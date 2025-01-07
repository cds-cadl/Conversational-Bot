import os
import subprocess

def generate_ssh_key(key_name="deploy_key"):
    """
    Generates an SSH key pair.
    :param key_name: Name of the key files to generate (default: deploy_key).
    :return: Path to the private and public key files.
    """
    private_key_path = os.path.expanduser(f"./{key_name}")
    public_key_path = f"{private_key_path}.pub"

    # Generate the SSH key pair
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-b", "2048", "-f", private_key_path, "-q", "-N", ""],
        check=True
    )
    return private_key_path, public_key_path

# Generate the SSH key pair
private_key_path, public_key_path = generate_ssh_key()

# Read the generated public key
with open(public_key_path, 'r') as pub_key_file:
    public_key_content = pub_key_file.read()

private_key_path, public_key_path, public_key_content
