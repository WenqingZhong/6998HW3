import json
import boto3

def lambda_handler(event, context):
    """
    main handler of events
    """
    notebook_instance_name = 'notebook-instance-wqzz'
    client = boto3.client('sagemaker')

    # start the notebook instance
    response = client.start_notebook_instance(NotebookInstanceName=notebook_instance_name)
    print("Start the notebook instance: ", response)
    
    print("retrain complete")

    return {
        'statusCode': 200,
        'body': json.dumps('Finished retraining the spam classifier.')
    }