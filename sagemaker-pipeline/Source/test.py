import boto3
import wget
import json
import numpy as np
import sys
import time

start = time.time()

endpoint_name = sys.argv[1]
configuration_file = sys.argv[2]

with open(configuration_file) as f:
    data = json.load(f)

commit_id = data["Parameters"]["CommitID"]
timestamp = data["Parameters"]["Timestamp"]


endpoint_name = endpoint_name + "-" + commit_id + "-" + timestamp

runtime = boto3.client('runtime.sagemaker') 

client = boto3.client('sagemaker-runtime')

sample_payload=b'low,low,5more,more,big,high'

response = client.invoke_endpoint(
    EndpointName='pipeline-car-evaluation',
    Body=sample_payload,
    ContentType='text/csv'
)

print('Our result for this payload is: {}'.format(response['Body'].read().decode('ascii')))

end = time.time()
seconds = end - start
seconds = repr(seconds)
print ("Time: " + seconds)