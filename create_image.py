from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image, ContainerImage


def main():
    # Get workspace
    ws = Workspace.get(name='mnist-azure', subscription_id='', resource_group='')

    # Get model
    model = Model(workspace=ws, name='mnist_tf_model', version=5)

    image_config = ContainerImage.image_configuration(runtime='python', execution_script='scoring.py',
                                                      conda_file='environment.yml')

    image = Image.create(name='mnist-tf', models=[model], image_config=image_config, workspace=ws)


if __name__ == '__main__':

    # Run main function
    main()
