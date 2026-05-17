import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

DOWNLOAD_DIR = "schematics"
FAILED_LOG = "failed_urls.txt"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

class McBuildScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        # Set a reasonable timeout for navigation
        self.page.set_default_timeout(60000)

    def stop(self):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def download_schematic(self, url):
            max_retries = 1
            for attempt in range(max_retries):
                try:
                    print(f"Processing: {url} (Attempt {attempt+1}/{max_retries})")
                    self.page.goto(url, timeout=60000)
                    self.page.wait_for_load_state("domcontentloaded")

                    # 1. Click the first "Download" button on the description page
                    # Selector: matches the green button linking to /download/schematic=...
                    # The button opens a new tab (target="_blank"), so we wait for the new page.
                    with self.context.expect_page(timeout=60000) as new_page_info:
                        # Wait for button to be clickable
                        dl_btn_1 = self.page.wait_for_selector("a.btn.btn-success[href*='/download/schematic=']", state="visible", timeout=60000)
                        dl_btn_1.click()
                    
                    download_page = new_page_info.value
                    download_page.wait_for_load_state("domcontentloaded")

                    # 2. Wait for the countdown on the interstitial page
                    # The second button has href starting with /download/ and contains ?k= usually (security token)
                    # We wait for this specific button to become visible on the NEW page.
                    final_download_selector = "a.btn.btn-success[href^='/download/'][href*='?k=']"
                    
                    # Wait for the button to solve the timer
                    dl_button = download_page.wait_for_selector(final_download_selector, state="visible", timeout=60000)
                    
                    if not dl_button:
                        download_page.close()
                        raise Exception("Download button did not appear after timer.")

                    # 3. Handle the download
                    with download_page.expect_download(timeout=60000) as download_info:
                        # Click the final button
                        dl_button.click()
                    
                    download = download_info.value
                    filename = download.suggested_filename
                    save_path = os.path.join(DOWNLOAD_DIR, filename)
                    
                    download.save_as(save_path)
                    print(f"  [SUCCESS] Saved to {save_path}")
                    
                    # Close the download page/tab
                    download_page.close()
                    return True

                except Exception as e:
                    print(f"  [WARN] Attempt {attempt+1} failed for {url}: {e}")
                    # If we opened a new page but failed, try to close it to clean up
                    try:
                        if 'download_page' in locals() and not download_page.is_closed():
                            download_page.close()
                    except:
                        pass
                    
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait a bit before retrying
                    else:
                        print(f"  [ERROR] All retries failed for {url}")
                        with open(FAILED_LOG, "a") as f:
                            f.write(f"{url} | Error: {e}\n")
                        return False

import sys

def main():
    input_file = "links11.txt"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("No URLs found in house.txt")
        return

    # Use headless=False so the user can see it working, or if debugging is needed.
    # Set to True for production speed if desired, but False is safer for avoiding some bot detection.
    scraper = McBuildScraper(headless=True) 
    try:
        scraper.start()
        print(f"Found {len(urls)} URLs. Starting scraper...")
        
        success_count = 0
        for i, url in enumerate(urls, 1):
             print(f"[{i}/{len(urls)}] ", end="")
             if scraper.download_schematic(url):
                 success_count += 1
             
             # Small delay between requests to be polite
             time.sleep(1) 
             
        print(f"\nScraping complete. {success_count}/{len(urls)} successful.")
        if os.path.exists(FAILED_LOG):
            print(f"Check {FAILED_LOG} for failed attempts.")

    finally:
        scraper.stop()

if __name__ == "__main__":
    main()
