input_file = "links.txt"
output_file = "links_cleaned.txt"

seen = set()
unique_links = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        link = line.strip()
        if link and link not in seen:
            seen.add(link)
            unique_links.append(link)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(unique_links))

print(f"Removed duplicates. {len(unique_links)} unique links saved.")
