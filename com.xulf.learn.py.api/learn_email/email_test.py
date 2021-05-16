__author__ = 'quiet road'

import smtplib
import unittest
from email.mime.text import MIMEText

class MailSenderText(unittest.TestCase):
    def test_send(self):
        # sender =MailSender("smtp.163.com", "fengge_developer@163.com", "fengge_developer",  "fenggekeji6")
        # sender =MailSender("smtp.163.com", "xuliufeng2010@163.com", "xuliufeng2010",  "skd798245503")
        sender =MailSender("smtp.exmail.qq.com", "xvliufeng@fengge.cn", "xvliufeng@fengge.cn",  "skd798245503")
        sender.add_receiver("xvliufeng@fengge.cn")
        sender.set_text_message("test_subject", "test_content")
        sender.send()

class MailSender(object):
    def __init__(self, mail_host, from_account, username, password):
        self.mail_host = mail_host
        self.from_account = from_account
        self.username = username
        self.password = password
        self.msg = None
        self.mimeText = None
        self.to_accounts = []

    def add_receiver(self, to_address):
        self.to_accounts.append(to_address)
    
    def set_text_message(self, subject, content):
        self.mimeText = MIMEText(content)
        self.mimeText['Subject'] = subject

    def send(self):
        self.prepare_msg()

        server = smtplib.SMTP()
        server.set_debuglevel(1)
        server.connect(self.mail_host)
        server.login(self.username, self.password)
        server.sendmail(self.from_account, self.to_accounts, self.msg)
        server.close()

    def prepare_msg(self):
        self.mimeText['From'] = self.from_account
        self.mimeText['To'] = ';'.join(self.to_accounts)
        self.msg = self.mimeText.as_string()
