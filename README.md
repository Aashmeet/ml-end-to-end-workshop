# ml-end-to-end-workshop
End to End machine learning process 



In this workshop, we’ll use Apache Spark MLLib for data processing using AWS Glue and reuse 
the data processing code during inference. We’ll use the Car Evaluation Data Set from 
UCI’s Machine Learning Repository.

Our goal is to predict the acceptability of a specific car, amongst the values of unacc, acc, good, and vgood. 

At the core, it is a classification problem, and we will train a machine learning model using Amazon SageMaker’s built-in 
XGBoost algorithm.
However, the dataset only contains six categorical string features – buying, maint, doors, persons, lug_boot, and safety and
XGBoost can only process data that is in numerical format.
Therefore we will pre-process the input data using SparkML StringIndexer followed by OneHotEncoder to convert
it to a numerical format. We will also apply a post-processing step on the prediction result using IndexToString to 
convert our inference output back to their original labels that correspond to the predicted condition of the car.

We’ll write our pre-processing and post-processing scripts once, and apply them for processing training data using AWS Glue. 
Then, we will serialize and capture these artifacts produced by AWS Glue to Amazon S3 using MLeap,
a common serialization format and execution engine for machine learning pipelines.
This is so the pre-processing steps can be reused during inference for real-time requests using the SparkML Serving container 
that Amazon SageMaker provides. 
Finally, we will deploy the pre-processing, inference, and post-processing steps in an inference pipeline and 
will execute these steps for each real-time inference request.


Data Engineering

Start a notebook instance and execute it 









Data Pipeline 

## Deployment Steps

####  Step 1. Create a GitHub OAuth Token
Create your token at [GitHub's Token Settings](https://github.com/settings/tokens), making sure to select scopes of **repo** and **admin:repo_hook**.  After clicking **Generate Token**, make sure to save your OAuth Token in a secure location. The token will not be shown again.

####  Step 2. Launch the Stack via CLI


`aws cloudformation create-stack --stack-name YOURSTACKNAME --template-body file:///home/ec2-user/environment/sagemaker-pipeline/CodePipeline/pipeline.yaml --parameters ParameterKey=Email,ParameterValue="youremailaddress@example.com" ParameterKey=GitHubToken,ParameterValue="YOURGITHUBTOKEN12345ab1234234" --capabilities CAPABILITY_NAMED_IAM`


<!-- [![Launch CFN stack](https://s3.amazonaws.com/stelligent-training-public/public/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#cstack=sn~DromedaryStack|turl~https://s3.amazonaws.com/aashmeet/master/d-master.json) -->

**Stack Assumptions:** The pipeline stack assumes the following conditions, and may not function properly if they are not met:
1. The pipeline stack name is less than 20 characters long
2

[![Launch CFN stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#cstack=sn~sagemaker-stack|turl~https://s3.amazonaws.com/sagemaker-pipeline-src/CodePipeline/pipeline.yaml)



####  Step 3. Test and Approve the Deployment
Once the deployment has passed automated QA testing, before proceeding with the production stage it sends an email notification (via SNS) for manual approval. At this time, you may run any additional tests on the endpoint before approving it to be deployed into production.
