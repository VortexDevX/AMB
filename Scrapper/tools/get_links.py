import re
import time
from playwright.sync_api import sync_playwright

OUTPUT_FILE = "scraped_links.txt"

def slugify(title):
    """
    Convert title to slug format used by mcbuild.org.
    - Lowercase
    - Replace spaces and special chars with dashes
    - Remove non-alphanumeric chars (except dashes)
    """
    # This is a best-effort slugify based on observation
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug

def scrape_links():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(60000)

        print("Navigating to https://mcbuild.org/schematics/...")
        page.goto("https://mcbuild.org/schematics/")
        time.sleep(2) # Wait for initial load

        # 1. Get all categories
        # Select all buttons in the nav-pills container
        # Based on investigation: .nav-pills button
        category_buttons = page.query_selector_all(".nav-pills button")
        category_ids = []
        for btn in category_buttons:
            t = btn.inner_text().strip()
            # Skip "Popular" if user says it doesn't show "more", OR handle it specifically.
            # User said: "popular category doesn't show anything more after clicking "more" remember"
            # We will still scrape what is there, but maybe not expect infinite scroll to work effectively if it's broken.
            # actually, let's treat it as a category we process.
            bid = btn.get_attribute("id")
            if bid:
                category_ids.append((t, bid))
        
        print(f"Found {len(category_ids)} categories: {[c[0] for c in category_ids]}")

        all_links = set()

        for cat_name, cat_id in category_ids:
            # Skip Popular explicitly if needed, but the loop handles it.
            if "Popular" in cat_name:
                 continue

            print(f"\nScraping category: {cat_name}")
            
            try:
                # Click category
                cat_btn = page.query_selector(f"#{cat_id}")
                if not cat_btn:
                     print(f"Could not find button #{cat_id}")
                     continue

                # Get the target container ID from data-bs-target
                target_container_selector = cat_btn.get_attribute("data-bs-target")
                if not target_container_selector:
                     print(f"No data-bs-target for {cat_name}")
                     continue
                
                # Ensure we are selecting ID only
                if target_container_selector.startswith('#'):
                     container_id = target_container_selector
                else:
                     container_id = f"#{target_container_selector}"

                cat_btn.click()
                time.sleep(5) # Wait for content
                
                # Handling "More" button loop
                while True:
                    # Scroll to bottom
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(3) 

                    # Check for "More" button SPECIFICALLY INSIDE THE ACTIVE CONTAINER
                    # This prevents clicking the hidden "Popular" button or others
                    more_btn = page.query_selector(f"{container_id} button.btn-more")
                    
                    if not more_btn:
                         print(" 'More' button not found in active container.")
                         break
                    
                    if not more_btn.is_visible():
                        print(" 'More' button found but not visible in active container. Attempting JS click...")
                    
                    # Check current card count
                    prev_count = len(page.query_selector_all(".card"))
                    
                    # Click more
                    try:
                        print(".", end="", flush=True)
                        more_btn.evaluate("e => e.click()")
                            
                        # Wait for new items to appear
                        time.sleep(5) # Increased wait to 5s

                        new_count = len(page.query_selector_all(".card"))
                        if new_count <= prev_count:
                             print(f" Item count ({new_count}) did not increase. Breaking.")
                             break
                             
                    except Exception as e:
                        print(f"Error clicking more: {e}")
                        break
                        
                print(" Done scrolling.")
            except Exception as e:
                print(f"Error processing category {cat_name}: {e}")

            # Extract links from all cards currently visible
            cards = page.query_selector_all(".card")
            cat_links_count = 0
            
            for card in cards:
                try:
                    link = None
                    
                    # 1. Try direct link first
                    a_tag = card.query_selector("a")
                    if a_tag:
                         href = a_tag.get_attribute("href")
                         if href and "mcbuild.org/schematics/" in href:
                             link = href
                    
                    # 2. If no direct link, try data-id for modal
                    if not link:
                        # Check on the card itself or any child
                        data_id = card.get_attribute("data-id")
                        data_title = card.get_attribute("data-title")
                        
                        if not data_id:
                            # Try finding a child with data-id
                            child_modal = card.query_selector("[data-id]")
                            if child_modal:
                                data_id = child_modal.get_attribute("data-id")
                                data_title = child_modal.get_attribute("data-title") # Might be missing on child
                        
                        if data_id:
                            # Clean up title for slug if missing
                            if not data_title:
                                # Try to find a title element
                                title_el = card.query_selector("h5, .card-title")
                                if title_el:
                                    data_title = title_el.inner_text().strip()
                                else:
                                    data_title = "unknown"
                            
                            slug = slugify(data_title)
                            link = f"https://mcbuild.org/schematics/{data_id}:{slug}"

                    if link:
                        if link not in all_links:
                            all_links.add(link)
                            cat_links_count += 1
                except Exception as e:
                    continue
            
            print(f"Found {cat_links_count} new links in {cat_name}")

        browser.close()

    print(f"\nTotal unique links found: {len(all_links)}")
    
    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link in sorted(all_links):
            f.write(link + "\n")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_links()
