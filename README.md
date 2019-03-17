# IFN_645
refer to folder IFN_645_draft
you may need to install graphviz in your computer
you may need to install pydot in your computer

2.1. For Windows users
For Windows users, simply download Anaconda and install it link. Choose the latest Python3 version for Windows.

Once you installed it, go to Start-Anaconda3-Anaconda Prompt. Type conda install seaborn to install Seaborn.

To install visualisation libraries for decision trees (practical 2), follow these steps:

Download graphviz for windows from http://www.graphviz.org/download/ and install it.
Add the graphviz bin folder to the PATH system environment variable (Example: "C:\Graphviz2.38\bin"). Tutorial here: http://windowsitpro.com/systems-management/how-can-i-add-new-folder-my-system-path.
Make sure git is installed in your system.
Go to Anaconda Prompt using start menu (Make sure to right click and select "Run as Administrator". We may get permission issues if Prompt as not opened as Administrator)
Execute the command: conda install graphviz
Execute the command: pip install git+https://github.com/nlhepler/pydot.git
Execute the command conda list and make sure pydot and graphviz modules are listed.
