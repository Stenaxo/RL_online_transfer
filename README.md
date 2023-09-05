# RL_online_transfer

The RL_online_transfer project serves as a platform to experiment with and implement deep learning solutions in the realm of online transfer learning. Leveraging the widely recognized Syn2Real database, this project has been devised to undertake two primary tasks that are pivotal in the field of machine learning:

1. **Offline Learning**: The script `cnn_off.py` facilitates offline machine learning model training. This stage utilizes the data available in the Syn2Real database to train the model based on traditional deep learning methodologies, thereby providing a strong foundation upon which online learning strategies can be built.

2. **Deployment with Online Transfer Learning**: Following the offline learning phase, the `cnn_on.py` script takes charge by deploying the pre-trained model and further adapting it using Online Transfer Learning techniques. This allows the model to continue learning and adapting over time, thereby improving accuracy and efficiency by incorporating new data and experiences.

By combining these two components, the RL_online_transfer project aims to not only create robust machine learning models but also facilitate a seamless transition into online learning environments where models can continue to grow and dynamically adapt.

To get started with this project, please follow the instructions below to install the necessary dependencies and set up your environment.


## Python Environment

A `requirements.txt` file is provided to help you install all the necessary dependencies. You can use the following command to set up your Python environment:

```sh
pip install -r requirements.txt
```

## Data Download
To download the dataset, follow the instructions below, ensuring to place it in the same directory as your code. Please adjust the cd command according to your specific directory structure.

```sh
cd ./data
wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  

wget http://csr.bu.edu/ftp/visda17/clf/test.tar
tar xvf test.tar

wget https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt
```

For more detailed information regarding the data or its utilization, please visit: [Reposit information](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

## Additional Notes

1. **Python Version**: Please ensure you are using a Python version that is compatible (we recommend 3.6 or later) before installing the dependencies from `requirements.txt`.
   
2. **Virtual Environment**: It is advisable to create and activate a virtual environment before installing the dependencies to avoid any conflicts with packages in the global Python environment. You can create a virtual environment using the commands below:
   
   ```sh
   python -m venv env
   source env/bin/activate  # On Unix or MacOS
   .\env\Scripts\activate   # On Windows
   ```

3. **Dataset Usage**: Be aware of any license or usage restrictions that might apply before utilizing the dataset.
4. **Project Structure**: Providing a brief overview of the project's directory structure, indicating where to place specific files (like data and scripts), would be beneficial.

## Usage

Once all the necessary dependencies have been installed, you are ready to utilize the project. Here is how you can go about it:

### Script Execution

To execute the scripts, use the `python3` command followed by the name of the script you wish to run (`cnn_off.py` for offline learning or `cnn_on.py` for online learning). Here are a couple of examples:

```bash
python3 cnn_off.py
python3 cnn_on.py
```

### Available Options
The scripts utilize argparse to handle command-line options. To view a list of all available options, you can use the -h or --help option as shown below:

```bash
python3 cnn_off.py -h
python3 cnn_on.py -h
```
This will display detailed help with all the options you can use with each script. Feel free to explore these options to further personalize your experience.
