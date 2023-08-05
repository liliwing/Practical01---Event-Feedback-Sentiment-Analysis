from google_service import GoogleService
from transformers import pipeline

if __name__ == '__main__':

    POSITIVE_RESPONSE = '''
    Please do not reply this email. This is a test email.
    Thank you. It's my pleasure.
    '''

    NEGATIVE_RESPONSE = '''
    Please do not reply this email. This is a test email.
    You suck.
    '''

    google_service = GoogleService()

    # Load sentiment model
    model = pipeline('sentiment-analysis', model="amazon-review")

    responses = google_service.get_responses()
    for response in responses:

        # Get email address from responses
        email = response['email']
        # Get feedback from responses
        feedback = response['feedback']
        # Make predictions. e.g. [{"label": "POSITIVE", "score": 0.99 }]
        prediction = model(feedback)[0]["label"]

        try:
            if prediction == "POSITIVE":
                google_service.send_email(email, POSITIVE_RESPONSE)
            else:
                google_service.send_email(email, NEGATIVE_RESPONSE)
        except HTTPError:
            break

    print("DONE!")
