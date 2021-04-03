# BSc ReadMe



## Requirements

Install python 3.8 -> https://www.python.org/downloads/
Install virtualenv via pip package manager -> https://docs.python-guide.org/dev/virtualenvs/ (scroll to virtualenv not pipenv)

In your terminal (I'm on windows so CMD)
```
$ > pip install --upgrade pip  
$ > pip install virtualenv
$ > cd myDir  
$ > virtualenv venv
$ > .\venv\Scripts\activate **(on Windows)**  
$ > source ./venv/bin/activate **(on Linux)**
```

I've created a requirements.txt containing all of the modules and their relative versions called **requirements.txt**.
Now run 
`$ >	pip install -r requirements.txt `