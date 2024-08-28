## Prepare your machine for NLP-Practice

We are going to use Visual Studio Code (VSCode) for our IDE. Then will prepare a Python Virtual Environment to isolate our working directory from our Operating System (OS) to make it more manageble.  
Please follow these steps:  
1. Download and install the [VSCode](https://code.visualstudio.com/download) for your OS.
1. Create a folder in your machine and you may name it as `NLP-practice`
1. Open the VSCode and from the menue choose `File` --> `Open Folder` and select the folder that you created (NLP-practice) 
1. From the left hand side toolbar, select the `Extensions`
   - Search for `Python` and install it. It may take some time.
1. From the top menue select `View` --> `Command Pallete..` 
   - You can use the shortcut `Ctrl`+`Shift`+`p` or `Cmd`+`Shift`+`p` 
1. Search for `Python: Create Environment...`
   - Click on it
   - Then select `Venv`. This will create a folder named `.venv` in your main folder.
1. From the top menu select Terminal and open a new one.
1. Now you should see a terminal with the command line starting with `(.venv)` in the begining of the prompt.
   - This shows that your environment is active and you can install the necessery python packages.
   - If you are using Windows, you may open a new `Command Prompt` Terminal from the drop down list in your terminal view.
1. When you are in the virtual environment, You can install the pakages either one-by-one using 
``` 
pip install <Your desired package name>
```
   - For example `pip install pandas`. This will only install `pandas` package in your environment.
   - You can also use the `requirements.txt` file to install all the neccessary packages by
   ```
   pip install -r requirements.txt
   ```

Now your Python environment is ready to be used to practice on the available NLP-practice codes in this repositpry.

Happy coding :-)