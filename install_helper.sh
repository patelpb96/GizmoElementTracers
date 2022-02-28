# Install utilities dependency
echo 'Installing utilities dependency...'
# Clone the repository
git clone git@bitbucket.org:awetzel/utilities.git
# Retrieve the setup.py script. This script must be in the directory *containing* the package it's run for (utilities).
cp ./utilities/setup.py ./
# Run the install script using pip
pip install -e .

# Install gizmo analysis
echo 'Installing gizmo_analysis...'
# Retrieve the setup.py script. This script must be in the directory *containing* the package it's run for (gizmo_analysis).
# This will overwrite the setup.py used for utilities
cp ./gizmo_analysis/setup.py ./
# Run the install script using pip
pip install -r gizmo_analysis/requirements.txt -e .

echo 'Installation complete!'