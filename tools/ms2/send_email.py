import smtplib
import ssl


def send_ms2_finished(email, link_address):
    with open('secrets/secret_email_key.txt') as f:
        pw = f.read().strip()
    #context=ssl.create_default_context()
    print("trying to log in to gmail")
    print("pw: {}".format(pw))
    with smtplib.SMTP_SSL(host="smtp.gmail.com", port=465) as server:
        server.login("epa.nta.app@gmail.com", pw)
        print("logged in to gmail")
        sender_email = "epa.nta.app@gmail.com"
        recipient_email = email
        message = "Your EPA NTA APP ms2 results are ready: {}".format(link_address)
        server.sendmail(sender_email, recipient_email, message)
        print("email sent to {}".format(recipient_email))
