# -*- coding=utf-8 -*-
from pyvi.pyvi import ViTokenizer,ViPosTagger
import requests



test_sent = "Đệ nhất phu nhân Mỹ Melania Trump cảm ơn Chelsea Clinton, con gái của cựu Tổng thống Bill Clinton và cựu Ngoại trưởng Hillary Clinton, vì đã lên tiếng bênh vực con trai 11 tuổi của bà trước những ý kiến trái chiều của dư luận."

url = "http://ai.topica.vn:9119/get_mlbka"


headers = {
    'cache-control': "no-cache",
    'postman-token': "dd327f89-2a5f-bf16-c115-590b590e32c3"
    }

response = requests.request("POST", url, data=test_sent, headers=headers)

tach_tu_anh_son = response.text

postag_as = ViPosTagger.postagging(tach_tu_anh_son)
y = zip(postag_as[0],postag_as[1])
print repr(y).decode('unicode-escape')

postaged_sent = ViPosTagger.postagging(ViTokenizer.tokenize(test_sent.decode('utf-8')))
x = zip(postaged_sent[0],postaged_sent[1])
print repr(x).decode('unicode-escape')