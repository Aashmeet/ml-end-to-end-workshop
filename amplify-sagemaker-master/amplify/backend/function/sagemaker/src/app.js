/*
Copyright 2017 - 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
    http://aws.amazon.com/apache2.0/
or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
*/

var express = require('express')
var bodyParser = require('body-parser')
var awsServerlessExpressMiddleware = require('aws-serverless-express/middleware')
var AWS = require('aws-sdk')

var sagemakerruntime = new AWS.SageMakerRuntime({ apiVersion: '2017-05-13' })
const customDomainAdaptorMiddleware = (req,res,next)=>{
    if (!! req.headers['x-apigateway-event']){
	    const event = JSON.parse(decodeURIComponent(req.headers['x-apigateway-event']));
	    const params = event.pathParameters||{};
   
	    let interpolated_resource = Object.keys(params)
	    	.reduce((acc, k)=>acc.replace('{'+k+'}', params[k]), event.resource)
	    // console.log(event.path, event.resource, interpolated_resource)
	    if ((!! event.path && !! interpolated_resource) && event.path != interpolated_resource){
		    req.url = req.originalUrl = interpolated_resource;
		    console.log(`rerouted ${event.path} -> ${interpolated_resource}`);
	    }
    }
    next()
}

// declare a new express app
var app = express()
app.use(bodyParser.json())
app.use(awsServerlessExpressMiddleware.eventContext())

//app.use(customDomainAdaptorMiddleware)
//app.use(express.static(path.join(__dirname, 'public')));


// Enable CORS for all methods
app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*")
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
 
  next()
});




/****************************
 * Sagemaker post method *
 ****************************/

app.get('/items', function(req, res) {
  var params = {
    Body: req.apiGateway.event.queryStringParameters.FinalPayload,
    EndpointName: req.apiGateway.event.queryStringParameters.EndpointName,
    ContentType: 'text/csv'
  };

  sagemakerruntime.invokeEndpoint(params, function(err, data) {
    if (err) {
     
      res.json({ error: err.stack, url: req.url, body: req.body })
      console.log(err, err.stack); // an error occurred
    }
    else {
    
      res.json({ success:  data.Body.toString('utf8')})
      console.log(data); // successful response
    }
  });


 
});

app.listen(3000, function() {
  console.log("App started")
});

// Export the app object. When executing the application local this does nothing. However,
// to port it to AWS Lambda we will create a wrapper around that will load the app from
// this file
module.exports = app


