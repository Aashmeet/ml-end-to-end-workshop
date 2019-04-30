#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING END TO END 
#  
# You deploy Inference Pipelines in Amazon SageMaker to execute a sequence of pre-processing, inference, and post-processing steps on real-time and batch inference requests. This makes it easy to build and deploy feature preprocessing pipelines with a suite of feature transformers available in the new SparkML and scikit-learn containers in Amazon SageMaker. You can write your data processing code once and reuse it for training and inference which provides consistency in your machine learning workflows and easier management of your models. You can deploy upto five steps in your inference pipeline and they all execute on the same instance so there is minimal latency impact. The same inference pipeline can be used for real-time and batch inferences.
# 
# # Part 1 - Data Engineering 

# ## Methodologies
# The Notebook consists of a few high-level steps:
# 
# * Using AWS Glue for executing the SparkML feature pre-processing and postprocessing job.
# * Using SageMaker XGBoost to train on the processed dataset produced by SparkML job.
# * Building an Inference Pipeline consisting of SparkML & XGBoost models for a realtime inference endpoint.

# # 1. Using AWS Glue for executing the SparkML job
# 
# We'll be running the SparkML job using [AWS Glue](https://aws.amazon.com/glue). AWS Glue is a serverless ETL service which can be used to execute standard Spark/PySpark jobs. Glue currently only supports `Python 2.7`, hence we'll write the script in `Python 2.7`.

# In[1]:


# Import SageMaker Python SDK to get the Session and execution_role
import sagemaker
from sagemaker import get_execution_role
sess = sagemaker.Session()
role = get_execution_role()
print(role[role.rfind('/') + 1:])


# ## 1.1 Adding AWS Glue as an additional trusted entity to this role
# This step is needed if you want to pass the execution role of this Notebook while calling Glue APIs as well without creating an additional **Role**. If you have not used AWS Glue before, then this step is mandatory.
# 
# If you have used AWS Glue previously, then you should have an already existing role that can be used to invoke Glue APIs. In that case, you can pass that role while calling Glue (later in this notebook) and skip this next step.
# 
# On the IAM dashboard, please click on **Roles** on the left sidenav and search for this Role. Once the Role appears, click on the **Role** to go to its Summary page. Click on the **Trust relationships** tab on the **Summary** page to add AWS Glue as an additional trusted entity.
# 
# Click on **Edit trust relationship** and replace the JSON with this JSON.
# 
# ```
# {
#   "Version": "2012-10-17",
#   "Statement": [
#     {
#       "Effect": "Allow",
#       "Principal": {
#         "Service": [
#           "sagemaker.amazonaws.com",
#           "glue.amazonaws.com"
#         ]
#       },
#       "Action": "sts:AssumeRole"
#     }
#   ]
# }
# ```
# 
# Once this is complete, click on **Update Trust Policy** and you are done.

# ## 1.2 Setup S3 bucket
# 
# First, we need to setup an S3 bucket within your account, and upload the necessary files to this bucket. To setup the bucket, we will run the first code block, labeled Setup S3 bucket. To run the cell while the code cell is selected, you can either press Shift and Return at the same time or select the Run button at the top of the Jupyter notebook.

# In[2]:


import boto3
import botocore
from botocore.exceptions import ClientError

from sagemaker import Session as Sess

# SageMaker session
sess = Sess()

# Boto3 session
session = boto3.session.Session()

s3 = session.resource('s3')
account = session.client('sts').get_caller_identity()['Account']
region = session.region_name
bucket_name = 'sagemaker-glue-process-{}-{}'.format("test", region)

try:
    if region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
except ClientError as e:
    error_code = e.response['Error']['Code']
    message = e.response['Error']['Message']
    if error_code == 'BucketAlreadyOwnedByYou':
        print ('A bucket with the same name already exists in your account - using the same bucket.')
        pass

print("\nSave this S3 bucket name for the rest of this process: {}".format(bucket_name))


# Make note of the S3 bucket name that was created here. If you are planning to follow along in the console, you will need this name for later.
# 
# ## 1.3 Upload files to S3
# 
# Now we need to upload the raw data and Glue processing script to S3. We can do that by running the code blocks in the notebook labeled Upload files to S3.

# In[3]:


get_ipython().run_cell_magic('bash', '', '\n# Download Raw data and Dependencies\nwget https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data\nwget https://s3-us-west-2.amazonaws.com/sparkml-mleap/0.9.6/python/python.zip\nwget https://s3-us-west-2.amazonaws.com/sparkml-mleap/0.9.6/jar/mleap_spark_assembly.jar')


# In[4]:


# Uploading the training data to S3
result = sess.upload_data(path='car.data', bucket=bucket_name, key_prefix='data')
print(result)
result = sess.upload_data(path='preprocessor.py', bucket=bucket_name, key_prefix='scripts')
print(result)
result = sess.upload_data(path='python.zip', bucket=bucket_name, key_prefix='scripts')
print(result)
result = sess.upload_data(path='mleap_spark_assembly.jar', bucket=bucket_name, key_prefix='scripts')
print(result)


# Your S3 bucket is now setup for our pipeline

# ## 1.4 Preprocessing using Apache Spark in AWS Glue
# 
# If you take a look at the data we downloaded, you’ll notice all of the fields are categorical data in string format, which XGBoost cannot natively handle. In order to utilize SageMaker’s XGBoost, we need to preprocess our data into a series of one hot encoded columns. Apache Spark provides preprocessing pipeline capabilities that we will utilize. 
# 
# Furthermore, to make our endpoint particularly useful, we also generate a post-processor in this script, which can convert our label indexes back to their original labels. All of these processor artifacts will be saved to S3 for SageMaker’s use later.
# 
# In this example, you downloaded our preprocessor.py script, and we recommend you take the time to explore how Spark pipelines are handled. Let’s take a look at the relevant part of the code where we define and fit our Spark pipeline:
# 
# ```
#     # Target label
#     catIndexer = StringIndexer(inputCol="cat", outputCol="label")
#     
#     labelIndexModel = catIndexer.fit(train)
#     train = labelIndexModel.transform(train)
#     
#     converter = IndexToString(inputCol="label", outputCol="cat")
# 
#     # Index labels, adding metadata to the label column.
#     # Fit on whole dataset to include all labels in index.
#     buyingIndexer = StringIndexer(inputCol="buying", outputCol="indexedBuying")
#     maintIndexer = StringIndexer(inputCol="maint", outputCol="indexedMaint")
#     doorsIndexer = StringIndexer(inputCol="doors", outputCol="indexedDoors")
#     personsIndexer = StringIndexer(inputCol="persons", outputCol="indexedPersons")
#     lug_bootIndexer = StringIndexer(inputCol="lug_boot", outputCol="indexedLug_boot")
#     safetyIndexer = StringIndexer(inputCol="safety", outputCol="indexedSafety")
#     
# 
#     # One Hot Encoder on indexed features
#     buyingEncoder = OneHotEncoder(inputCol="indexedBuying", outputCol="buyingVec")
#     maintEncoder = OneHotEncoder(inputCol="indexedMaint", outputCol="maintVec")
#     doorsEncoder = OneHotEncoder(inputCol="indexedDoors", outputCol="doorsVec")
#     personsEncoder = OneHotEncoder(inputCol="indexedPersons", outputCol="personsVec")
#     lug_bootEncoder = OneHotEncoder(inputCol="indexedLug_boot", outputCol="lug_bootVec")
#     safetyEncoder = OneHotEncoder(inputCol="indexedSafety", outputCol="safetyVec")
# 
#     # Create the vector structured data (label,features(vector))
#     assembler = VectorAssembler(inputCols=["buyingVec", "maintVec", "doorsVec", "personsVec", "lug_bootVec", "safetyVec"], outputCol="features")
# 
#     # Chain featurizers in a Pipeline
#     pipeline = Pipeline(stages=[buyingIndexer, maintIndexer, doorsIndexer, personsIndexer, lug_bootIndexer, safetyIndexer, buyingEncoder, maintEncoder, doorsEncoder, personsEncoder, lug_bootEncoder, safetyEncoder, assembler])
# 
#     # Train model.  This also runs the indexers.
#     model = pipeline.fit(train)
# ```
# 
# This snippet defines both our preprocessor and postprocessor. The preprocessor converts all the training columns from categorical labels into a vector of one-hot encoded columns, while the post-processor converts our label index back to a human readable string.
# 
# In addition, it may be helpful to examine the code which allows us to serialize and store our Spark pipeline artifacts in the MLeap format. Because the Spark framework was designed around batch use cases, we need to use MLeap here. MLeap serializes SparkML Pipelines and provides run time for deploying for real-time, low latency use cases. Amazon SageMaker has launched a SparkML Serving container that uses MLEAP to make it easy to use. Let’s look at the code below:
# 
# ```
#     # Serialize and store via MLeap  
#     SimpleSparkSerializer().serializeToBundle(model, "jar:file:/tmp/model.zip", predictions)
#     
#     # Unzipping as SageMaker expects a .tar.gz file but MLeap produces a .zip file.
#     import zipfile
#     with zipfile.ZipFile("/tmp/model.zip") as zf:
#         zf.extractall("/tmp/model")
# 
#     # Writing back the content as a .tar.gz file
#     import tarfile
#     with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
#         tar.add("/tmp/model/bundle.json", arcname='bundle.json')
#         tar.add("/tmp/model/root", arcname='root')
# 
#     s3 = boto3.resource('s3')
#     file_name = args['s3_model_bucket_prefix'] + '/' + 'model.tar.gz'
#     s3.Bucket(args['s3_model_bucket']).upload_file('/tmp/model.tar.gz', file_name)
# 
#     os.remove('/tmp/model.zip')
#     os.remove('/tmp/model.tar.gz')
#     shutil.rmtree('/tmp/model')
#     
#     # Save postprocessor
#     SimpleSparkSerializer().serializeToBundle(converter, "jar:file:/tmp/postprocess.zip", predictions)
# 
#     with zipfile.ZipFile("/tmp/postprocess.zip") as zf:
#         zf.extractall("/tmp/postprocess")
# 
#     # Writing back the content as a .tar.gz file
#     import tarfile
#     with tarfile.open("/tmp/postprocess.tar.gz", "w:gz") as tar:
#         tar.add("/tmp/postprocess/bundle.json", arcname='bundle.json')
#         tar.add("/tmp/postprocess/root", arcname='root')
# 
#     file_name = args['s3_model_bucket_prefix'] + '/' + 'postprocess.tar.gz'
#     s3.Bucket(args['s3_model_bucket']).upload_file('/tmp/postprocess.tar.gz', file_name)
# 
#     os.remove('/tmp/postprocess.zip')
#     os.remove('/tmp/postprocess.tar.gz')
#     shutil.rmtree('/tmp/postprocess')
# ```
# 
# You’ll notice we unzip this archive and re-archive it into a tar.gz file that SageMaker recognizes.
# 
# To run our Spark pipelines within SageMaker, we are going to utilize our notebook instance.  within the SageMaker notebook, you can run the cell labeled Create and run AWS Glue Preprocessing Job, which is in the following cell. This cell will define the job in Glue, run the job, and monitor the status until the job has completed.
# 
# 
# ## 1.5 Create and run AWS Glue Preprocessing Job (###SKIP THIS IF MANUAL JOB HAS RUN SUCCESSFULLY)
# 
# Next we'll be creating Glue client via Boto so that we can invoke the `create_job` API of Glue. `create_job` API will create a job definition which can be used to execute your jobs in Glue. The job definition created here is mutable. While creating the job, we are also passing the code location as well as the dependencies location to Glue.
# 
# The  job will be executed by calling `start_job_run` API. This API creates an immutable run/execution corresponding to the job definition created above. We will require the `job_run_id` for the particular job execution to check for status. We'll pass the data and model locations as part of the job execution parameters.
# 
# Finally we will check for the job status to see if it has `succeeded`, `failed` or `stopped`. Once the job is succeeded, we have the transformed data into S3 in CSV format which we can use with XGBoost for training. If the job fails, you can go to [AWS Glue console](https://us-west-2.console.aws.amazon.com/glue/home), click on **Jobs** tab on the left, and from the page, click on this particular job and you will be able to find the CloudWatch logs (the link under **Logs**) link for these jobs which can help you to see what exactly went wrong in the job execution.

# In[ ]:


### Create and run AWS Glue Preprocessing Job
###SKIP THIS IF MANUAL JOB HAS RUN SUCCESSFULLY
# Define the Job in AWS Glue
glue = boto3.client('glue')

try:
    glue.get_job(JobName='preprocessing-cars')
    print("Job already exists, continuing...")
except glue.exceptions.EntityNotFoundException:
    response = glue.create_job(
        Name='preprocessing-cars',
        Role=role,
        Command={
            'Name': 'glueetl',
            'ScriptLocation': 's3://{}/scripts/preprocessor.py'.format(bucket_name)
        },
        DefaultArguments={
            '--s3_input_data_location': 's3://{}/data/car.data'.format(bucket_name),
            '--s3_model_bucket_prefix': 'model',
            '--s3_model_bucket': bucket_name,
            '--s3_output_bucket': bucket_name,
            '--s3_output_bucket_prefix': 'output',
            '--extra-py-files': 's3://{}/scripts/python.zip'.format(bucket_name),
            '--extra-jars': 's3://{}/scripts/mleap_spark_assembly.jar'.format(bucket_name)
        }
    )

    print('{}\n'.format(response))

# Run the job in AWS Glue
try:
    job_name='preprocessing-cars'
    response = glue.start_job_run(JobName=job_name)
    job_run_id = response['JobRunId']
    print('{}\n'.format(response))
except glue.exceptions.ConcurrentRunsExceededException:
    print("Job run already in progress, continuing...")

    
# Check on the job status
import time

job_run_status = glue.get_job_run(JobName=job_name,RunId=job_run_id)['JobRun']['JobRunState']
while job_run_status not in ('FAILED', 'SUCCEEDED', 'STOPPED'):
    job_run_status = glue.get_job_run(JobName=job_name,RunId=job_run_id)['JobRun']['JobRunState']
    print (job_run_status)
    time.sleep(30)


# In summary, we have now preprocessed our data into a training and validation set, with one-hot encoding for all of the string values. We have also serialized a preprocessor and post-processor into the MLeap format, so that we can reuse these pipelines in our endpoint later. The next step is to train a Machine Learning model. We will be using Amazon SageMaker’s built-in XGBoost for this.
# 
# 
# #  Lab 2 -Machine Learning Process
# 
# 
# 
# ## Training an Amazon SageMaker XGBoost Model
# 
# Now that we have our data preprocessed in a format that XGBoost recognizes, we can run a simple training job to train a classifier model on our data. We can run this entire process in our Jupyter notebook. Run the following cell, labeled Run Amazon SageMaker XGBoost Training Job. This will run our XGBoost training job in Amazon SageMaker, and monitor the progress of the job. Once the job is ‘Completed’, you can move on to the next cell.
#  
# This will train the model on the preprocessed data we created earlier. After a few minutes, usually less than 5, the job should complete successfully, and output our model artifacts to the S3 location we specified. Once this is done, we can deploy an inference pipeline that consists of pre-processing, inference and post-processing steps.
# 
# ## 2.1 Run Amazon SageMaker XGBoost Training Job

# In[6]:


### Run Amazon SageMaker XGBoost Training Job

from sagemaker.amazon.amazon_estimator import get_image_uri

import random
import string
import time

# Get XGBoost container image for current region
training_image = get_image_uri(region, 'xgboost', repo_version="latest")

# Create a unique training job name
training_job_name = 'xgboost-cars-'+''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

# Create the training job in Amazon SageMaker
sagemaker = boto3.client('sagemaker')
response = sagemaker.create_training_job(
    TrainingJobName=training_job_name,
    HyperParameters={
        'early_stopping_rounds ': '5',
        'num_round': '10',
        'objective': 'multi:softmax',
        'num_class': '4',
        'eval_metric': 'mlogloss'

    },
    AlgorithmSpecification={
        'TrainingImage': training_image,
        'TrainingInputMode': 'File',
    },
    RoleArn=role,
    InputDataConfig=[
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://{}/output/train'.format(bucket_name),
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None',
            'RecordWrapperType': 'None',
            'InputMode': 'File'
        },
        {
            'ChannelName': 'validation',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://{}/output/validation'.format(bucket_name),
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None',
            'RecordWrapperType': 'None',
            'InputMode': 'File'
        },
    ],
    OutputDataConfig={
        'S3OutputPath': 's3://{}/xgb'.format(bucket_name)
    },
    ResourceConfig={
        'InstanceType': 'ml.m4.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 1
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600
    },)

print('{}\n'.format(response))

# Monitor the status until completed
job_run_status = sagemaker.describe_training_job(TrainingJobName=training_job_name)['TrainingJobStatus']
while job_run_status not in ('Failed', 'Completed', 'Stopped'):
    job_run_status = sagemaker.describe_training_job(TrainingJobName=training_job_name)['TrainingJobStatus']
    print (job_run_status)
    time.sleep(30)


# ## 2.2 Deploying an Amazon SageMaker Endpoint utilizing your data processing artifacts
# 
# Now that we have a set of model artifacts, we can set up an inference pipeline that executes sequentially in Amazon SageMaker. We start by setting up a Model, which will point to all of our model artifacts, then we setup an Endpoint configuration to specify our hardware, and finally we can stand up an Endpoint. With this endpoint, we will pass the raw data and no longer need to write pre-processing logic in our application code. The same pre-processing steps that ran for training can be applied to inference input data for better consistency and ease of management.
# 
# Deploying a model in SageMaker requires two components:
# 
# * Docker image residing in ECR.
# * Model artifacts residing in S3.
# 
# **SparkML**
# 
# For SparkML, Docker image for MLeap based SparkML serving is provided by SageMaker team. For more information on this, please see [SageMaker SparkML Serving](https://github.com/aws/sagemaker-sparkml-serving-container). MLeap serialized SparkML model was uploaded to S3 as part of the SparkML job we executed in AWS Glue.
# 
# **XGBoost**
# 
# For XGBoost, we will use the same Docker image we used for training. The model artifacts for XGBoost was uploaded as part of the training job we just ran.
# 
# 
# ## Create SageMaker Endpoint with pipeline

# In[10]:


### Create SageMaker endpoint with pipeline
from botocore.exceptions import ClientError

# Image locations are published at: https://github.com/aws/sagemaker-sparkml-serving-container
sparkml_images = {
    'us-west-1': '746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'us-west-2': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'us-east-1': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'us-east-2': '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ap-northeast-1': '354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ap-northeast-2': '366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ap-southeast-1': '121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ap-southeast-2': '783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ap-south-1': '720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'eu-west-1': '141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'eu-west-2': '764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'eu-central-1': '492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'ca-central-1': '341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-sparkml-serving:2.2',
    'us-gov-west-1': '414596584902.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-sparkml-serving:2.2'
}



try:
    sparkml_image = sparkml_images[region]

    response = sagemaker.create_model(
        ModelName='pipeline-car-evaluation',
        Containers=[
            {
                'Image': sparkml_image,
                'ModelDataUrl': 's3://{}/model/model.tar.gz'.format(bucket_name),
                'Environment': {
                    'SAGEMAKER_SPARKML_SCHEMA': '{"input":[{"type":"string","name":"buying"},{"type":"string","name":"maint"},{"type":"string","name":"doors"},{"type":"string","name":"persons"},{"type":"string","name":"lug_boot"},{"type":"string","name":"safety"}],"output":{"type":"double","name":"features","struct":"vector"}}'
                }
            },
            {
                'Image': training_image,
                'ModelDataUrl': 's3://{}/xgb/{}/output/model.tar.gz'.format(bucket_name, training_job_name)
            },
            {
                'Image': sparkml_image,
                'ModelDataUrl': 's3://{}/model/postprocess.tar.gz'.format(bucket_name),
                'Environment': {
                    'SAGEMAKER_SPARKML_SCHEMA': '{"input": [{"type": "double", "name": "label"}], "output": {"type": "string", "name": "cat"}}'
                }

            },
        ],
        ExecutionRoleArn=role
    )

    print('{}\n'.format(response))
    
except ClientError:
    print('Model already exists, continuing...')


try:
    response = sagemaker.create_endpoint_config(
        EndpointConfigName='pipeline-car-evaluation',
        ProductionVariants=[
            {
                'VariantName': 'DefaultVariant',
                'ModelName': 'pipeline-car-evaluation',
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge',
            },
        ],
    )
    print('{}\n'.format(response))

except ClientError:
    print('Endpoint config already exists, continuing...')


try:
    response = sagemaker.create_endpoint(
        EndpointName='pipeline-car-evaluation',
        EndpointConfigName='pipeline-car-evaluation',
    )
    print('{}\n'.format(response))

except ClientError:
    print("Endpoint already exists, continuing...")


# Monitor the status until completed
endpoint_status = sagemaker.describe_endpoint(EndpointName='pipeline-car-evaluation')['EndpointStatus']
while endpoint_status not in ('OutOfService','InService','Failed'):
    endpoint_status = sagemaker.describe_endpoint(EndpointName='pipeline-car-evaluation')['EndpointStatus']
    print(endpoint_status)
    time.sleep(30)


# After a few minutes, Amazon SageMaker will have created an endpoint utilizing all three of the provided containers on a single instance. When the endpoint is invoked with a payload, the output of the earlier containers is passed as the input to the later containers, until the payload reaches its final output.
# 
# In this example, the raw, string categories are sent to our preprocessing SparkML serving container and run through a Spark pipeline to one hot encode the features. Then the one hot encoded data is sent to our XGBoost container, where our model makes a prediction to an index. The index is then fed to our post-processing MLeap container, with a Spark model artifact, which converts the index back to its original label string, which is returned to the client. These are the exact same steps you used for pre-processing training data and it was only necessary to write the code once.
# 
# ## 2.3 Testing the Endpoint
# 
# Once the Amazon SageMaker endpoint is InService, we can test it with the code cell labeled Invoke the Endpoint. If successful, this should return one of the following values: `unacc`, `acc`, `good`, `vgood`.
# 
# ## 2.4 Invoke the Endpoint
# 
# ### Invoking the newly created inference endpoint with a payload to transform the data
# Now we will invoke the endpoint with a valid payload that SageMaker SparkML Serving can recognize. There are three ways in which input payload can be passed to the request:
# 
# * Pass it as a valid CSV string. In this case, the schema passed via the environment variable will be used to determine the schema. For CSV format, every column in the input has to be a basic datatype (e.g. int, double, string) and it can not be a Spark `Array` or `Vector`.
# 
# * Pass it as a valid JSON string. In this case as well, the schema passed via the environment variable will be used to infer the schema. With JSON format, every column in the input can be a basic datatype or a Spark `Vector` or `Array` provided that the corresponding entry in the schema mentions the correct value.
# 
# * Pass the request in JSON format along with the schema and the data. In this case, the schema passed in the payload will take precedence over the one passed via the environment variable (if any).
# 
# In this case, we will pass it as a valid CSV string.

# In[11]:


### Invoke the Endpoint
client = boto3.client('sagemaker-runtime')

sample_payload=b'low,low,5more,more,big,high'

response = client.invoke_endpoint(
    EndpointName='pipeline-car-evaluation',
    Body=sample_payload,
    ContentType='text/csv'
)

print('Our result for this payload is: {}'.format(response['Body'].read().decode('ascii')))


# ## 4 Clean up your AWS environment
# When you are done with this experiment, make sure to delete your SageMaker endpoint to avoid incurring unexpected costs. You can do this from the AWS Console by going to Services, Amazon SageMaker, Inference, and Endpoints. Select pipeline-xgboost under Endpoints. In the upper-right, select Delete. This will remove the endpoint from your AWS account. You will also want to make sure to stop your Notebook instance.
# 
# A more extensive cleanup can be done from your Notebook instance by running the code cell labeled Environment cleanup, seen below.
# 
# ## Environment cleanup -POST INFERENCE

# In[ ]:


### Environment cleanup




# ## References
# 
# Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

# In[ ]:




