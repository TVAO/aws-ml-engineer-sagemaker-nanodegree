import json
import boto3
import base64

s3 = boto3.resource('s3') # Using client does not give access to bucket 

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"] 
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    s3.Bucket(bucket).download_file(key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

import json
import base64
import boto3 

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2022-03-05-14-01-41-157"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    """A function to invoke model endpoint and return prediction"""
    print(event) 

    # Decode the image data
    image = base64.b64decode(event['image_data'])
    
    # QUESTION: Could not get commented out code to work. How can I solve this more elegantly? 
    # import sagemaker
    # from sagemaker.serializers import IdentitySerializer
    # from sagemaker.predictor import Predictor 
    # Instantiate a Predictor
    # predictor = Predictor(ENDPOINT)
    # For this model the IdentitySerializer needs to be "image/png"
    # predictor.serializer = IdentitySerializer("image/png")
    # Make a prediction:
    # inferences = predictor.predict(data=image)
    # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')]
    
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences']
        }
        # 'body': json.dumps(event)
    }

import json

THRESHOLD = .7

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any (inference >= THRESHOLD for inference in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ValueError(f"No prediction satisfies threshold of {THRESHOLD}")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }