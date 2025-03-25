import subprocess
import tempfile
import os

def mermaid_to_image(mermaid_str, output_path):
    """
    Convert a Mermaid string to an image using mermaid-cli.

    :param mermaid_str: The Mermaid diagram text (string)
    :param output_path: Path (with extension .png or .svg) to write the resulting image
    """
    # Create a temporary file for the Mermaid code
    with tempfile.NamedTemporaryFile(suffix=".mmd", delete=False) as tmp:
        tmp.write(mermaid_str.encode("utf-8"))
        tmp_name = tmp.name

    # Run mermaid-cli to convert the .mmd file to an image
    subprocess.run([
        "mmdc",
        "-i", tmp_name,
        "-o", output_path
    ], check=True)

    # Clean up the temporary .mmd file
    os.remove(tmp_name)

if __name__ == "__main__":
    # Sample Mermaid code
    diagram = """
    graph TD;
        A --> B;
        B --> C;
        C --> D;
        A --> D;
    """

    # Convert Mermaid string to 'diagram.png'
    mermaid_to_image(diagram, "diagram.png")
    print("Generated diagram.png")
