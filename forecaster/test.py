import json
import time

def handler(event, context):
    # TODO implement
    
    path = event['1']
    method = event['2']
    
    response_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": 'GET, POST, PUT, DELETE, OPTIONS',
        "content-type":"application/json",
    }
    
    # default response
    response = 'Nothing to see here ... this is not the motivation you are looking for. Would have been nice if there was a better error message here, huh.'
    # response = json.dumps(event, indent=2)
    
    if path == '/motivational_message' and method == 'GET':
        motivation = {'response': {'type': 'motivator', 'content': {"it's in here somewhere": '... but not here', 'here-it-is:': 'GET PORKING!!!'}}}
        response = json.dumps(motivation)
        
    elif path == '/how_do_i/get_going.json' and method == 'GET':
        motivation = {'response': {'type': 'motivator-my-way', 'content': {"it's in here somewhere": '... but not here', 'here-it-is:': '<div class="motivation"><h2>You got to ...</h2><video class="media-url__media" loop="" autoplay="" muted="" playsinline="" width="40%"><source src="https://i.gifer.com/8Pba.mp4" itemprop="contentUrl" type="video/mp4"></video><h2>PUSH IT</h2></div>'}}}
        response = json.dumps(motivation)
        
    elif path == '/motivational_message' and method == 'POST':
        if 'PUSH-IT-REAL-GOOD' in event['headers']:
            time.sleep(4)
            response = 'YEAH! All done.  This request takes a while, right? '

    elif path == '/mirror':
        response = json.dumps(event, indent=4)
    
    return {
        'statusCode': 200,
        'headers': response_headers,
        'body': response
    }
