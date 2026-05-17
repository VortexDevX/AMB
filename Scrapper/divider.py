import os

def split_links(input_file, start_index=6, chunk_size=200):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        # Read all lines and strip whitespace
        links = [line.strip() for line in f if line.strip()]

    total_links = len(links)
    print(f"Found {total_links} links in {input_file}.")

    file_index = start_index
    for i in range(0, total_links, chunk_size):
        chunk = links[i:i + chunk_size]
        output_filename = f"links{file_index}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(chunk) + '\n')
        
        print(f"Created {output_filename} with {len(chunk)} links.")
        file_index += 1

if __name__ == "__main__":
    split_links('links_left.txt')
