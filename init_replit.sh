function getDeltaFolder()
{
    find . -name "*delta_*" -type d
}

folder=$(getDeltaFolder)
mv $folder/* $folder/.* .
rm -r $folder/
pip install -r requirements.txt
pip3 install torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
