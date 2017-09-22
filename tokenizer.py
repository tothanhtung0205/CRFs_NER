# -*- coding=utf-8 -*-
import requests
from pyvi.pyvi import ViPosTagger,ViTokenizer


def get_tokenizer(sent):

    sent = sent.encode('utf-8')
    url = "http://ai.topica.vn:9119/get_mlbka"
    headers = {
    'cache-control': "no-cache",
    'postman-token': "dd327f89-2a5f-bf16-c115-590b590e32c3"
    }

    response = requests.request("POST", url, data=sent, headers=headers)

    return response.text


# x = get_tokenizer(u"Tôi là học sinh")
# y = ViTokenizer.tokenize(u"tôi là học sinh")
#
# print x
# print y
