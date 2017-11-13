rm -rf tipster/
git clone https://github.com/abednego1979/tipster.git
sudo python tools.py
cd tipster
git add .
git commit -m 'patch0413'
git push -u origin master
cd ..