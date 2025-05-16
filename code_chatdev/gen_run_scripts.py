import re

def update_script(file_path, new_file_path, old_string, new_string):
    # Read the content of the file
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Regex to find "--config test_1 &" and replace "test_1" with "test_2"
        # pattern = r'(--config\s+)' + re.escape(old_string) + r'(\s*&)'
        # replaced_content = re.sub(pattern, r'\1' + new_string + r'\2', content)

        pattern = r'(--config\s+)' + re.escape(old_string)
        replaced_content = re.sub(pattern, r'\1' + new_string, content)

        # Write the updated content back to the file
        with open(new_file_path, 'w') as file:
            file.write(replaced_content)

        print(f"File '{file_path}' has been updated successfully.")
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except IOError as e:
        print(f"An error occurred while handling the file: {e}")


# Strings to replace
old_string = 'test_1'
new_string = 'test_'


reference_scripts = "SRDD/data/data_ChatDev_format_100_test1.sh"
new_reference_scripts = "SRDD/data/data_ChatDev_format_100_test{}.sh"

# Update the script
for i in range(2, 10):
    update_script(reference_scripts, new_reference_scripts.format(i), old_string, new_string+str(i))
