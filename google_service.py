from __future__ import print_function

import base64
from apiclient import discovery
from email.mime.text import MIMEText
from httplib2 import Http
from oauth2client import client, file, tools
from pprint import pprint
from requests import HTTPError

class GoogleService:

    def __init__(self):

        # Allow actions
        SCOPES = [
            "https://www.googleapis.com/auth/forms.responses.readonly",
            "https://www.googleapis.com/auth/gmail.send"
        ]
        DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

        # Authenication
        store = file.Storage('token.json')
        creds = None
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
            creds = tools.run_flow(flow, store)

        # Create Google Form API
        self.gform_service = discovery.build('forms', 'v1', http=creds.authorize(
            Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

        # Create Gmail API
        self.gmail_service = discovery.build('gmail', 'v1', credentials=creds)

    def get_responses(self):
        form_id = '1qfv7wfzqtJS-KtHpYTp-77ZHHxnMaFZ_iWRV__jLeSE'
        result = self.gform_service.forms().responses().list(formId=form_id).execute()
        responses = result["responses"]

        pretty_responses = []
        for response in responses:
            # Get email address from responses
            email = response['respondentEmail']
            # Get feedback from responses
            answer = response['answers']['3b68337d']['textAnswers']['answers'][0]["value"]
            pretty_responses.append({'email' : email, 'feedback' : answer})

        return pretty_responses

    def send_email(self, email, content):
        message = MIMEText(content)
        message['to'] = email
        message['subject'] = 'Test Email'
        create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
        try:
            self.gmail_service.users().messages().send(userId="me", body=create_message).execute()
        except HTTPError as error:
            print(F'An error occurred: {error}')
            raise HTTPError
