################################################################
# Options
################################################################

# Which bitbucket user to pull from?
BITBUCKET_USER=zhafen
# BITBUCKET_USER=awetzel

# Optional installs
INSTALL_HALO_ANALYSIS=true
INSTALL_JUPYTERLAB=true

################################################################
# Installation Script
################################################################

# Install utilities dependency
echo ''
echo 'Installing utilities dependency...'
echo ''
# Clone the repository
git clone git@bitbucket.org:$BITBUCKET_USER/utilities.git
# Retrieve the setup.py script. This script must be in the directory *containing* the package it's run for (utilities).
cp ./utilities/setup.py ./
# Run the install script using pip
pip install -e .

# Install utilities dependency
if $INSTALL_HALO_ANALYSIS
then
	echo ''
	echo 'Installing halo_analysis dependency...'
	echo ''
	# Clone the repository
	git clone git@bitbucket.org:$BITBUCKET_USER/halo_analysis.git
	# Retrieve the setup.py script. This script must be in the directory *containing* the package it's run for (utilities).
	cp ./halo_analysis/setup.py ./
	# Run the install script using pip
	pip install -r halo_analysis/requirements.txt -e .
fi

# Install gizmo analysis
echo ''
echo 'Installing gizmo_analysis...'
echo ''
# Retrieve the setup.py script. This script must be in the directory *containing* the package it's run for (gizmo_analysis).
# This will overwrite the setup.py used for utilities
cp ./gizmo_analysis/setup.py ./
# Run the install script using pip
pip install -r gizmo_analysis/requirements.txt -e .

# Install jupyter lab
if $INSTALL_JUPYTERLAB
then
	echo ''
	echo 'Installing jupyter-lab...'
	echo ''
	pip install jupyterlab
fi

echo ''
echo 'Installation complete!'