
"""aaa"""
import os
import codecs
import json
import glob
import time
import threading
from collections import defaultdict, Counter
import itertools
import gc
from io import BytesIO
import base64
import csv
import unicodedata
import shutil
import configparser
import errno
import math
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeResult
import requests
import numpy as np
import pymupdf
from PIL import Image, ImageFile
from watchfiles import watch

ImageFile.LOAD_TRUNCATED_IMAGES = True
allpage_count = 1

config_ini = configparser.ConfigParser()
#config_ini.read(os.getcwd() + '\\ocr_key3.ini', 'UTF-8')
config_ini.read('C:\\Users\\USER\\Desktop\\NAS_TEST\\ocr_key3.ini', 'UTF-8') #絶対パス

# 指定したiniファイルが存在しない場合、エラー発生
if not os.path.exists('C:\\Users\\USER\\Desktop\\NAS_TEST\\ocr_key3.ini'):  #絶対パス
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'C:\\Users\\USER\\Desktop\\NAS_TEST\\ocr_key3.ini')  #絶対パス

document_intelligence_client = DocumentIntelligenceClient(endpoint=config_ini['DEFAULT']['endpoint'], credential=AzureKeyCredential(config_ini['DEFAULT']['api_key']), api_version="2024-11-30")

API_KEY = config_ini['DEFAULT']['google_key']
direction = config_ini['DEFAULT']['direction']
output = config_ini['DEFAULT']['output']
plan = config_ini['DEFAULT']['plan']
code = config_ini['DEFAULT']['code']

watch_directory = config_ini['DEFAULT']['watch_directory']
_dpi = int(config_ini['DEFAULT']['_dpi'])

def pil_to_base64(img1, format2="png"):
    """
    aaa
    """
    buffer = BytesIO()
    img1.save(buffer, format2)
    img_str = base64.b64encode(buffer.getvalue())
    return img_str


def text_detection(zimage_path):
    """
    aaa
    """
    api_url = f'https://vision.googleapis.com/v1/images:annotate?key={API_KEY}'

    # base64 文字列 (jpeg) に変換する。
    img1 = Image.open(zimage_path)
    image_content = pil_to_base64(img1, format2="png")

    req_body = json.dumps({
        'requests': [{
            'image': {
                'content':  image_content.decode('utf-8')  # base64でエンコードしたものjsonにするためdecodeする
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION' #'TEXT_DETECTION' 'DOCUMENT_TEXT_DETECTION'
            }],
            "imageContext": {
                "languageHints": ["ja"]
            },
        }]
    })
    res = requests.post(api_url, data=req_body, timeout=15)
    return res.json()


def google(aifile):
    """
    aaa
    """
    if not os.path.isfile(aifile.replace('.jpeg', '_google.json').replace('.jpg', '_google.json')
                          .replace('.png', '_google.json').replace('.tiff', '_google.json')
                          .replace('.tif', '_google.json')):
        res_json = text_detection(aifile)
        while len(res_json) == 0:
            time.sleep(1)
        if len(res_json["responses"][0]) == 0:
            with open(aifile.replace('.jpeg', '_google.json').replace('.jpg', '_google.json')
                      .replace('.png', '_google.json').replace('.tiff', '_google.json')
                      .replace('.tif', '_google.json'), "w", encoding='utf-8') as js:
                js.write("")
            #continue
        else:
            res_text = res_json["responses"][0]["fullTextAnnotation"]
            with open(aifile.replace('.jpeg', '_google.json').replace('.jpg', '_google.json')
                      .replace('.png', '_google.json').replace('.tiff', '_google.json')
                      .replace('.tif', '_google.json'), "w", encoding='utf-8') as js:
                json.dump(res_text, js, indent=4, ensure_ascii=False)

    return

def azure(aifile):
    """
    aaa
    """
    with open(aifile, "rb") as file:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-read", #prebuilt-layout
            analyze_request=file,
            features=[DocumentAnalysisFeature.LANGUAGES],
            content_type="application/octet-stream",
        )
        # ステータスが完了になるまでポーリング
        while not poller.done():
            # print(f"{datetime.now()}: Waiting...")
            time.sleep(2)
        # 結果を取得
        result: AnalyzeResult = poller.result()
        # AnalyzeResultオブジェクトを辞書に変換
        result_json = json.dumps(result.as_dict(), indent=4, ensure_ascii=False)
        data1 = json.loads(result_json)

        with codecs.open(aifile.replace('.jpeg', '_azure.json').replace('.jpg', '_azure.json').replace('.png', '_azure.json').replace('.tiff', '_azure.json').replace('.tif', '_azure.json'), 'w', 'utf-8') as f2:
            json.dump(data1, f2, ensure_ascii=False, indent=2)

    return

def round_to_nearest_multiple(rx, base):
    """
    aaa
    """
    if int(base) == 0:
        base = 1.0

    return base * round(rx / base)

def del_chara(zlist):
    """
    aaa
    """
    if len(zlist) > 0:
        #座標が重複する文字を削除 ⑴同文字、⑵信頼度比較
        for iz in range(0, 5, 1):
            if  iz <= 2:
                zlist = sorted(zlist, key=lambda x: (int(x[22]), ord(x[0][0]), int(x[7])))
            else:
                zlist = sorted(zlist, key=lambda x: (int(x[22]), ord(x[0][0]), int(x[8])))

            for lz in range(0, len(zlist), 1):
                if lz >= 1 and lz+1 < len(zlist):
                    if zlist[lz][22] == zlist[lz-1][22] and zlist[lz][24] == zlist[lz-1][24]:
                        za = [min(zlist[lz-1][3], zlist[lz-1][5]), min(zlist[lz-1][4], zlist[lz-1][6]), max(zlist[lz-1][3], zlist[lz-1][5]), max(zlist[lz-1][4], zlist[lz-1][6])]
                        zb = [min(zlist[lz][3], zlist[lz][5]), min(zlist[lz][4], zlist[lz][6]), max(zlist[lz][3], zlist[lz][5]), max(zlist[lz][4], zlist[lz][6])]
                        iou_ab = iou(za, zb)
                        # print(zlist[lz], zlist[lz-1], " iou =", iou_ab)
                        if iou_ab > 0.15: # and (unicodedata.east_asian_width(zlist[lz-1][0][0]) in 'FWA' or unicodedata.east_asian_width(zlist[lz][0][0]) in 'FWA')
                            if zlist[lz][0] == zlist[lz-1][0] and not zlist[lz-1][0] in 'tw/':#⑴ # www.https//: を除く
                                del zlist[lz-1]
                            else:
                                if zlist[lz-1][11] <= zlist[lz][11] and zlist[lz][24] == 'google':#⑵
                                    del zlist[lz]
                                elif zlist[lz-1][11] > zlist[lz][11] and zlist[lz][24] == 'google':#⑵
                                    del zlist[lz-1]
                                else:
                                    continue

    return zlist

def half_width_division(zlist):
    """
    aaa
    """
    zzlist =[]
    for xz in zlist:
        tc = len(xz[0])
        tz = xz[0]
        if tc > 1:
            for z in range(tc):
                wmx = 0
                wmy = 0
                zx0 = 0
                zy0 = 0
                zx1 = 0
                zy1 = 0
                if xz[13] == "vertical":
                    c_width = int(xz[9])
                    c_height = int(xz[10]/tc)
                    if xz[12] == 'r000' :
                        zx1 = int(xz[3]+c_width)
                        zy0 = int(xz[4]+c_height*z)
                        zy1 = zy0+c_height
                        wmx = xz[7] #int((zy0+zy1)*0.5)
                        wmy = int((zy0+zy1)*0.5) #xz[8]
                    elif xz[12] == 'r180' :
                        zx1 = int(xz[3]+c_width)
                        zy0 = int(xz[4]+c_height*z)
                        zy1 = zy0+c_height
                        wmx = xz[7] #int((zy0+zy1)*0.5)
                        wmy = int((zy0+zy1)*0.5) #xz[8]
                    elif xz[12] == 'r090':
                        zx1 = int(xz[3]+c_width)
                        zy0 = int(xz[4]+c_height*z)
                        zy1 = zy0+c_height
                        wmx = xz[7] #int((zy0+zy1)*0.5)
                        wmy = int((zy0+zy1)*0.5) #xz[8]
                    elif xz[12] == 'r270':
                        zx1 = int(xz[3]-c_width)
                        zy0 = int(xz[4]-c_height*z)
                        zy1 = zy0-c_height
                        wmx = xz[7] #int((zy0+zy1)*0.5)
                        wmy = int((zy0+zy1)*0.5) #xz[8]
                    zzlist.append([tz[z], xz[1], z, xz[3], zy0, xz[5], zy1,
                                ceil(int(wmx),30), ceil(int(wmy),30), c_width, c_height,
                                xz[11], xz[12], xz[13], xz[14], xz[15], xz[16], xz[17], xz[18], xz[19], xz[20], xz[21], xz[22], xz[23], xz[24]])

                elif xz[13] == "horizontal":
                    c_width = int(xz[9]/tc)
                    c_height = int(xz[10])
                    if xz[12] == 'r000':
                        zy1 = int(xz[4]+c_height)
                        zx0 = int(xz[3]+c_width*z)
                        zx1 = zx0+c_width-1
                    elif xz[12] == 'r090':
                        zy1 = int(xz[4]+c_height)
                        zx0 = int(xz[5]-c_width*z)
                        zx1 = zx0-c_width+1
                    elif xz[12] == 'r270':
                        zy1 = int(xz[4]-c_height)
                        zx0 = int(xz[3]+c_width*z)
                        zx1 = zx0+c_width
                    elif xz[12] == 'r180':
                        print('NG')
                        zy1 = int(xz[4]-c_height)
                        zx0 = int(xz[3]-c_width*z)
                        zx1 = zx0-c_width
                    wmx = int((zx0+zx1)*0.5) #xz[7]
                    wmy = xz[8]
                    zzlist.append([tz[z], xz[1], z, zx0, xz[4], zx1, zy1,
                                ceil(int(wmx),30), ceil(int(wmy),30), c_width, c_height,
                                xz[11], xz[12], xz[13], xz[14], xz[15], xz[16], xz[17], xz[18], xz[19], xz[20], xz[21], xz[22], xz[23], xz[24]])
        else:
            zzlist.append(xz)

    return zzlist

def iou(za, zb):
    """
    a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    """
    ax_mn, ay_mn, ax_mx, ay_mx = za[0], za[1], za[2], za[3]
    bx_mn, by_mn, bx_mx, by_mx = zb[0], zb[1], zb[2], zb[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    zw = max(0, abx_mx - abx_mn + 1)
    zh = max(0, aby_mx - aby_mn + 1)
    intersect = zw*zh

    ziou = intersect / (a_area + b_area - intersect)
    return ziou

def get_east_asian_width_count(ztext):
    """
    aaa
    """
    count = 0
    for c in ztext:
        if unicodedata.east_asian_width(c) in 'FWA':
            count += 2
        else:
            count += 1
    return count


def near_point(xw, yw, xz, yz):
    """
    @return 対象値に最も近い値
    """
    if len(xz) == 0:
        return

    distance = (xz - xw) ** 2 + (yz - yw) ** 2
    iz = np.argmin(distance)
    # print(xz[iz], yz[iz], iz)
    return iz


def characters_comp(azlist, gzlist):
    """
    aaa
    """
    mxlist = []
    zaddlist = []
    # mxlist = [[i, r[7], r[8], r[22]] for i, r in enumerate(gzlist)]
    mxlist = np.array([[r[7], r[22]] for r in gzlist])
    mylist = np.array([[r[8], r[22]] for r in gzlist])

    #az中心値の近似値をgzlistから検出し、相違文字であれば座標の重なり比較を行い、重複がなければ文字を追加
    for az in azlist:
        mxlist2 = mxlist[mxlist[:, 1] == az[22]]
        mylist2 = mylist[mylist[:, 1] == az[22]]
        mxlist2 = mxlist[:, 0]
        mylist2 = mylist[:, 0]

        index = near_point(az[7], az[8], mxlist2, mylist2)
        if index is None:
            index = 0
        if gzlist[index][0] == az[0]:
            continue
        else:
            #index±2文字の同字または重なりを調べ、重複がなければ追加
            idiff = 0

            for zb in range(-2, 2, 1):
                if index+zb == len(gzlist):
                    break
                if gzlist[index+zb][0] == az[0]:
                    idiff = 1
                    break
                elif gzlist[index+zb][0] in '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳' and unicodedata.east_asian_width(az[0]) in 'HNa':
                    idiff = 1
                    break

                # elif unicodedata.east_asian_width(gzlist[index+zb][0]) in 'HNa' and gzlist[index+zb][0] in az[0]:
                #     idiff = 1
                #     break

            if idiff == 0:

                za = [min(az[3], az[5]), min(az[4], az[6]), max(az[3], az[5]), max(az[4], az[6])]
                if az[0] in '。｡、､．.\'･,()\"' and not az[23] == 'red':
                    hw = int(min(abs(az[5]-az[3]), abs(az[6]-az[4]))*4.0)
                    za[0] -= hw
                    za[1] -= hw
                    za[2] += hw
                    za[3] += hw

                zg = [min(gzlist[index][3], gzlist[index][5]), min(gzlist[index][4], gzlist[index][6]), max(gzlist[index][3], gzlist[index][5]), max(gzlist[index][4], gzlist[index][6])]
                iou_ab = iou(za, zg)
                # if az[0] == '発':
                # print(gzlist[index], az, iou_ab)
                if iou_ab > 0.3 or (az[0] in '。｡、､．.\'･,()\"' and gzlist[index][0] in '。｡、､．.\'･,()\"' and iou_ab > 0.1):
                    if gzlist[index][11] > az[11]:
                        continue
                    else:
                        gzlist.insert(index, az)
                        gzlist[index][1] = gzlist[index+1][1]
                        if az[13] == 'horizontal':
                            gzlist[index][1] = gzlist[index+1][1]
                            gzlist[index][4] = gzlist[index+1][4]
                            gzlist[index][6] = gzlist[index+1][6]
                            gzlist[index][8] = gzlist[index+1][8]
                            gzlist[index][14] = gzlist[index+1][14]
                            gzlist[index][15] = gzlist[index+1][15]
                        elif az[13] == 'vertical':
                            gzlist[index][1] = gzlist[index+1][1]
                            gzlist[index][3] = gzlist[index+1][3]
                            gzlist[index][5] = gzlist[index+1][5]
                            gzlist[index][7] = gzlist[index+1][7]
                            gzlist[index][14] = gzlist[index+1][14]
                            gzlist[index][15] = gzlist[index+1][15]
                        del gzlist[index+1]
                else:
                    # print(gzlist[index], az)
                    zaddlist.append(az)

    return gzlist, zaddlist

def pdf_separate(ifile1, doc1, page_count1):
    """
    aaa
    """
    for page_number in range(page_count1):
        doc3 = pymupdf.open()
        doc3.insert_pdf(doc1, from_page=page_number, to_page=page_number)
        doc3.save(ifile1.replace('.pdf', '') + '_' + str(page_number+1).zfill(4) + '★.pdf')  # ファイルとして保存します

    return

def ceil(number, multiple):
    """
    aaa
    """
    return math.ceil(number / multiple) * multiple

def sort_vlist1(zlist):
    """
    aaa
    """
    if len(zlist)>0:

        #row[10]<50本文
        xy = [[ceil(int(row[14]),20), ceil(int(row[15]),20)+20, ceil(int(row[16]),20)-10, ceil(int(row[17]),20)-10, ceil(int(row[16])-int(row[14]), 20)] for row in zlist if row[13]=='horizontal' and row[1]==1 and int(row[10])<=50]
        # xy = list(map(list, set(map(tuple, xy))))
        # #row[10]>=50見出し
        xy2 = [[ceil(int(row[14]),20), ceil(int(row[15]),20), ceil(int(row[16]),20)-10, ceil(int(row[17]),20)] for row in zlist if row[13]=='horizontal' and row[1]==1 and int(row[10])>50]
        # xy2 = list(map(list, set(map(tuple, xy2))))

        #各文字をx0座標でソート
        xy.sort(key=lambda x: x[0])

        # 本文x0の差が80であればグループ化
        x_groups = defaultdict(list)
        x_continuity_threshold = 90  # x軸方向の6連続性閾値
        for zline in xy:
            zx0, zy0, zx1, zy1, zwidth = zline
            added = False
            for key in list(x_groups.keys()):
                if abs(zx0 - key) <= x_continuity_threshold:
                    x_groups[key].append([zx0, zy0, zx1, zy1, zwidth])
                    added = True
                    break
            if not added:
                x_groups[zx0].append([zx0, zy0, zx1, zy1, zwidth])

        y_continuity_threshold = 50  # x軸方向の6連続性閾値
        blocks = []
        current_block = None
        for key in x_groups.keys():
            x_groups[key] = sorted(x_groups[key], key=lambda x: x[3])
            for xrect in x_groups[key]:
                x0_r, y0_r, x1_r, y1_r, zwidth = xrect

                #widthより大きい場合の処理を入れる20250425
                if current_block is None:
                    current_block = {'x0s': [x0_r], 'y0s': [y0_r], 'x1s': [x1_r], 'y1s': [y1_r], 'width': [zwidth]}
                else:
                    # 同じx座標かつy軸の重なりを考慮（回転後の座標に合わせてロジック調整）
                    if abs(x0_r - current_block['x0s'][-1]) <= x_continuity_threshold:
                        last_y0_of_block = max(current_block['y0s'])
                        if y1_r > last_y0_of_block + y_continuity_threshold:
                            # y方向に大きな隙間がある場合、新しいブロック開始
                            min_y0 = max(current_block['y0s'])
                            min_x0 = min(current_block['x0s'])
                            max_y1 = min(current_block['y1s'])
                            x1_counts = Counter(current_block['x1s'])
                            most_common_x1 = x1_counts.most_common(1)[0][0]

                            w_counts = Counter(current_block['width'])
                            most_common_w = w_counts.most_common(1)[0][0]

                            blocks.append((min_x0, min_y0, most_common_x1, max_y1))
                            if len(blocks) >= 2 and blocks[-1]==blocks[-2]:
                                blocks.pop(-1)

                            # 新しいブロック開始
                            current_block = {'x0s': [x0_r], 'y0s': [y0_r], 'x1s': [x1_r], 'y1s': [y1_r], 'width': [zwidth]}
                        else:
                            # y方向に重なりがある場合、統合
                            current_block['x0s'].append(x0_r)
                            current_block['y0s'].append(y0_r)
                            current_block['x1s'].append(x1_r)
                            current_block['y1s'].append(y1_r)
                            current_block['width'].append(zwidth)
                    else:
                        # x座標が異なる場合、ブロックを確定
                        min_y0 = max(current_block['y0s'])
                        min_x0 = min(current_block['x0s'])
                        max_y1 = min(current_block['y1s'])
                        x1_counts = Counter(current_block['x1s'])
                        most_common_x1 = x1_counts.most_common(1)[0][0]

                        w_counts = Counter(current_block['width'])
                        most_common_w = w_counts.most_common(1)[0][0]

                        blocks.append((min_x0, min_y0, most_common_x1, max_y1))

                        if len(blocks) >= 2 and blocks[-1]==blocks[-2]:
                            blocks.pop(-1)

                        # 新しいブロック開始
                        current_block = {'x0s': [x0_r], 'y0s': [y0_r], 'x1s': [x1_r], 'y1s': [y1_r], 'width': [zwidth]}

            # 最後のブロックを確定
            if current_block:
                min_y0 = max(current_block['y0s'])
                min_x0 = min(current_block['x0s'])
                max_y1 = min(current_block['y1s'])
                x1_counts = Counter(current_block['x1s'])
                most_common_x1 = x1_counts.most_common(1)[0][0]
                w_counts = Counter(current_block['width'])
                most_common_w = w_counts.most_common(1)[0][0]

                blocks.append((min_x0, min_y0, most_common_x1, max_y1))

            # print(blocks)

        #zlistに本文ブロック座標を代入
        for i, z in enumerate(zlist):
            for b, rect in enumerate(blocks):
                x0, y0, x1, y1 = rect
                if min(x0,x1)-10<=int(zlist[i][7])<=max(x0,x1)+10 and min(y0,y1)-10<=int(zlist[i][8])<=max(y0,y1)+10 and zlist[i][13]=='horizontal':
                    zlist[i][18], zlist[i][19], zlist[i][20], zlist[i][22], zlist[i][23] = x0, y0, x1, b, y1
                    break

        iz = 0
        mx0list = np.array([r[0] for r in blocks])
        my0list = np.array([r[1] for r in blocks])
        mx1list = np.array([r[2] for r in blocks])
        my1list = np.array([r[3] for r in blocks])

        # 見出しを本文に挿入
        for i, z in enumerate(zlist):
            if  zlist[i][18]==0 and zlist[i][13]=='horizontal':
                for ix0, iy0, ix1, iy1  in xy2:
                    if ceil(int(zlist[i][14]),20)==ix0 and ceil(int(zlist[i][15]),20)==iy0 and ceil(int(zlist[i][16]),20)-10==ix1 and ceil(int(zlist[i][17]),20)==iy1:
                        distance = (mx0list-ix0)**2+(my0list-iy0)**2+(mx1list-ix1)**2+(my1list-iy1)**2
                        iz = np.argmin(distance)
                        # print(int(iz))
                        zlist[i][18], zlist[i][19], zlist[i][20], zlist[i][22], zlist[i][23] = ix0, iy1, ix1, int(iz), iy0
                        m=i+1
                        if len(zlist)+1==m:
                            continue

                        while len(zlist)>m and ceil(int(zlist[m][14]),20)==ix0 and ceil(int(zlist[m][15]),20)==iy0 and ceil(int(zlist[m][16]),20)-10==ix1 and ceil(int(zlist[m][17]),20)==iy1:
                            zlist[m][18], zlist[m][19], zlist[m][20], zlist[m][22], zlist[m][23] = ix0, iy1, ix1, int(iz), iy0
                            m=m+1
                        break

        # 0の場合、本文に挿入
        for i, z in enumerate(zlist):
            if  zlist[i][18]==0 and zlist[i][13]=='horizontal':
                distance = (mx0list-int(zlist[i][14]))**2+(my0list-int(zlist[i][15]))**2+(mx1list-int(zlist[i][16]))**2+(my1list-int(zlist[i][17]))**2
                iz = np.argmin(distance)
                #print(iz)
                zlist[i][18], zlist[i][19], zlist[i][20], zlist[i][22], zlist[i][23] = ceil(int(zlist[i][14]),20), ceil(int(zlist[i][17]),20)-10, ceil(int(zlist[i][16]),20)-10, iz, ceil(int(zlist[i][15]),20)+15

        zhlist = []
        zvlist = []
        for zx in zlist:
            if zx[13]=='horizontal':
                zhlist.append(zx)
            else:
                zvlist.append(zx)
        zvlist = sorted(zvlist, key=lambda x: (int(x[21]), int(x[1]), int(x[2])))
        zhlist = sorted(zhlist, key=lambda x: (int(x[22]), int(x[15]), int(x[1]), int(x[2])))
        zlist = zvlist+zhlist
        return zlist

def group_by_custom_interval(lst, fix):
    grouped = []
    current_group = []
    if len(lst) > 0:
        fix=lst[0][1]-lst[0][0]
        current_max =  lst[0][0]+fix*0.9
        
        for num in lst:
            if num[0] < current_max:
                current_group.append(num[0])
            else:
                grouped.append(current_group)
                current_group = [num[0]]
                fix=num[1]-num[0]
                current_max = num[0]+fix*0.9

        if current_group:
            grouped.append(current_group)

    return grouped

def sort_vlist2(zlist):
    """
    aaa
    """
    if len(zlist)>0:

        hlist = []
        vlist = []
        for zx in zlist:
            if zx[13] == 'vertical':
                vlist.append(zx)
            elif zx[13] == 'horizontal':
                hlist.append(zx)

        xy = np.array([(ceil(r[14], 60), ceil(r[16], 60)) for r in hlist if r[1] == 1 and r[5] != r[16] and int(r[9])<=55])
        #見出し2025/4/25
        # xy2 = np.array([(ceil(r[14], 60), ceil(r[16], 60)) for r in hlist if r[1] == 1 and r[5] != r[16] and int(r[9])>55])

        # 辞書を使って0列の値ごとに1列の値の出現回数をカウント
        value_counts = defaultdict(Counter)
        for row in xy:
            value_counts[row[0]][row[1]] += 1

        # 最も多い値を取得
        y_line = []
        for key, counter in value_counts.items():
            most_common_value = counter.most_common(1)[0][0]
            y_line.append([int(key), int(most_common_value)])

        # 昇順に並び替え
        y_line = sorted(y_line, key=lambda x: (int(x[0]), int(x[1])))

        y_lst = group_by_custom_interval(y_line, 260)

        # 置換処理
        for i, r in enumerate(hlist):
            for group in y_lst: #y_lst
                if min(group)<=hlist[i][22]<=max(group):
                    hlist[i][18] = group[0]
                    break

        iz = 0
        
        mx0list = np.array([r[0] for r in y_lst])

        # 見出しを本文に挿入
        for i, z in enumerate(hlist):
            if  hlist[i][1]==1 and hlist[i][18]==0 and hlist[i][13]=='horizontal':
                distance = (mx0list-int(hlist[i][14]))**2
                iz = np.argmin(distance)
                hlist[i][18] = int(mx0list[int(iz)])
                m=i+1
                if len(hlist)==m:
                        continue

                while hlist[i][14]==hlist[m][14] and hlist[i][15]==hlist[m][15] and hlist[i][16]==hlist[m][16] and hlist[i][17]==hlist[m][17]:
                    hlist[m][18] = int(mx0list[int(iz)])
                    m=m+1
                    if len(hlist)==m:
                        continue

        hlist = sorted(hlist, key=lambda x: (int(x[18]), int(x[17]), int(x[1]), int(x[2])))
        vlist = sorted(vlist, key=lambda x:(x[21], x[1], x[2]))
        zlist = vlist+hlist
    
    return zlist

def sort_vlist3(zlist):
    """
    aaa
    """
    if len(zlist)>0:

        if len(zlist)>0:
            # 文字幅のNumPy配列を作成
            my_array = np.array([r[9] for r in zlist])
            # 各値の出現頻度を取得  
            values, counts = np.unique(my_array, return_counts=True)
            # 最も頻繁に出現する文字幅を取得
            most_frequent_value = values[counts.argmax()]
            print(most_frequent_value)

            #市民タイムスは文字サイズ×1.7、市民タイムス以外は文字サイズ×1.5
            for zl in range(0, len(zlist), 1):
                if zlist[zl][14]<=100: #欄外先頭処理
                    zlist[zl][22] = 0
                else:
                    zlist[zl][22] = round_to_nearest_multiple(zlist[zl][14], most_frequent_value*1.7)

            coordinates = np.array([(round_to_nearest_multiple(r[14], most_frequent_value*1.7),
                                    round_to_nearest_multiple(r[16], most_frequent_value*1.7))
                                    for r in zlist if r[1] == 1 and r[9] <= most_frequent_value*1.7]) #1.5固定

            # 辞書を使って0列の値ごとに1列の値の出現回数をカウント
            value_counts = defaultdict(Counter)
            for row in coordinates:
                value_counts[row[0]][row[1]] += 1

            # 最も多い値を取得
            y_line = []
            for key, counter in value_counts.items():
                most_common_value = counter.most_common(1)[0][0]
                y_line.append([key, most_common_value])

            # 昇順に並び替え
            y_line = sorted(y_line, key=lambda x: (int(x[1]), int(x[0])))

            #市民タイムス、以外ともに文字サイズ×6.0
            y_lst = group_by_custom_interval(y_line, most_frequent_value*6.0)

            # 置換処理
            for zi, r in enumerate(zlist):
                for group in y_lst: #y_lst
                    if zlist[zi][22] in group:
                        zlist[zi][22] = group[0]
                        break

            zlist = sorted(zlist, key=lambda x: (int(x[22]), int(x[15]), int(x[14]), int(x[1]), int(x[2])))
            zx = 0
            for zi in range(0, len(zlist), 1):
                if zi > 0 and abs(zlist[zi-1][15]-zlist[zi][15])>=most_frequent_value*2:
                    zx = zx + 1
                    zlist[zi][22] = str(int(zlist[zi][22])) + '-' + str(zx).zfill(3)
                else:
                    zlist[zi][22] = str(int(zlist[zi][22])) + '-' + str(zx).zfill(3)

            #段落ごとに'horizontal'が多い場合は'horizontal'段落の順位を並び替え
            y_line = []
            zx = 0
            for zi in range(zx, len(zlist)):
                if zi == 0 or zlist[zi][22] == zlist[zi-1][22]:
                    y_line.append(zlist[zi][13])
                else:
                    if len(y_line) == 0:
                        y_line = []
                        zx = zi
                        continue
                    else:
                        # 出現回数をカウント
                        counter = Counter(y_line)
                        # 最も多い値を取得
                        most_common_value, most_common_count = counter.most_common(1)[0]
                        if most_common_value == 'horizontal':
                            zlist[zx:zi] = sorted(zlist[zx:zi], key=lambda x: (int(x[21]), int(x[1]), int(x[2])))
                            for zl in range(zx, zi):
                                zlist[zl][22] = 'h' + str(zlist[zl][22])
                        y_line = []
                        zx = zi
            if most_common_value == 'horizontal':
                zlist[zx:zi] = sorted(zlist[zx:zi], key=lambda x: (int(x[21]), int(x[1]), int(x[2])))
                for zl in range(zx, zi+1):
                    zlist[zl][22] = 'h' + str(zlist[zl][22])

    zhlist = []
    zvlist = []
    for zx in zlist:
        if 'h' in zx[22]:
            zhlist.append(zx)
        else:
            zvlist.append(zx)
    # print(int(x[22][1:-4]), int(x[22][-3]), int(x[15]), int(x[14]))

    zhlist = sorted(zhlist, key=lambda x: (int(x[22][1:-4]), int(x[22][-3]), int(x[21]) , int(x[1]), int(x[2])))
    zlist = zhlist + zvlist

    return zlist

action_type_map = {
    1: "Added",
    2: "Modified",
    3: "Deleted",
}


#読込みファイルはPDF固定
addlist = []
ss = ''
for changes in watch(watch_directory + '\\01_input'):
    for action, spath in changes:

        # 変更のタイプが「更新」の場合、別の関数を実行する
        if action_type_map.get(action) == "Added":

            addlist = glob.glob(watch_directory + '\\01_input\\*.pdf') + glob.glob(watch_directory + '\\01_input\\*.tif') + glob.glob(watch_directory + '\\01_input\\*.tiff') + glob.glob(watch_directory + '\\01_input\\*.png') + glob.glob(watch_directory + '\\01_input\\*.jpg') + glob.glob(watch_directory + '\\01_input\\*.jpeg')
            for ifile in addlist:


                dname = watch_directory + '\\03_work'
                if os.path.isfile(dname + '\\' + os.path.basename(ifile).replace('.*', '_ocr.png')) is False:

                    if '.pdf' in ifile:
                        #pdf読み込み
                        doc = pymupdf.open(ifile, filetype="pdf")

                        #1頁目取得
                        page = doc[0]

                        #pdf回転初期化
                        page.remove_rotation()
                        doc.saveIncr()

                        #複数頁の場合、頁番号を付与して分割保存
                        if doc.page_count >= 2:
                            #pdf頁数取得
                            allpage_count = doc.page_count
                            page_count = doc.page_count
                            pdf_separate(ifile, doc, doc.page_count)
                            doc.close()
                            os.remove(ifile)
                            break

                        #背景画像になければ'06_back_image'に格納
                        back_image = ifile.replace('01_input','06_back_image')
                        if not os.path.exists(back_image):
                            shutil.copy(ifile, ifile.replace('01_input','06_back_image'))

                        #画像がない白頁等はTLTファイルとして'02_output'に格納
                        if len(doc.get_page_images(0)) == 0:
                            doc.save(watch_directory + '\\02_output\\' + os.path.basename(ifile).replace('.pdf', '_TLT.pdf'), garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True)

                            if os.path.isfile(watch_directory + '\\01_input\\completed\\' + os.path.basename(ifile)) is True:
                                os.remove(watch_directory + '\\01_input\\completed\\' + os.path.basename(ifile))
                            shutil.move(ifile, watch_directory + '\\01_input\\completed')

                            continue
                    else:
                        time.sleep(0.5)
                        img = pymupdf.open(ifile)  # open pic as document
                        pdfbytes = img.convert_to_pdf()  # make a PDF stream
                        img.close()  # no longer needed
                        doc = pymupdf.open("pdf", pdfbytes)  # open stream as PDF
                        page = doc[0]
                        page.set_rotation(270) #OCRイメージを90度回転
                        page.remove_rotation()
                        allpage_count = 1
                        page_count = 1

                    #ページ全体の画像をimgに読み込む
                    img = page.get_pixmap(dpi=_dpi)
                    image_width, image_height = img.width, img.height

                    if img.n == 4:
                        mode = 'RGBA'
                    else:
                        mode = 'RGB'

                    #OCR用画像を出力
                    ocrname = dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + "_ocr.jpg"
                    img.pil_save(ocrname, optimize=True, dpi=(_dpi, _dpi))

                #ocr処理
                if plan == "A シングル":
                    print('A')
                    thread1 = threading.Thread(target=azure, args=(ocrname,)) #1
                    thread1.start() #2
                    thread1.join() #3

                if plan == "A シングル":

                    #azure処理
                    alist = []

                    #一時フォルダにあらかじめjsonを格納しておく
                    azure_name = dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '_ocr_azure.json'
                    azure_size = 0
                    azure_size = os.path.getsize(azure_name)
                    # time.sleep(1)
                    while azure_size != os.path.getsize(azure_name):
                        time.sleep(1)
                        azure_size = os.path.getsize(azure_name)

                    #azure
                    if os.path.isfile(azure_name) is True:
                        json_open = open(azure_name, 'r', encoding='utf-8')
                        json_load = json.load(json_open)
                        file_size = os.path.getsize(azure_name)
                        if json_load['content']== '' and file_size == 0:
                            shutil.copyfile(azure_name.replace('_ocr_azure.json',  '.pdf'), azure_name.replace('_ocr_azure.json',  '_OCR.pdf'))
                        else:

                            #行取得
                            plist = []
                            cl = 0
                            for line in json_load['pages'][0]['lines']:
                                cl = cl+1

                                offset = line["spans"][0]["offset"]
                                length = line["spans"][0]["length"]

                                x0, y0, x1, y1 = line["polygon"][0], line["polygon"][1], line["polygon"][4], line["polygon"][5]

                                # if x0 > x1:
                                #     x0, x1 = line["polygon"][4], line["polygon"][0]

                                # if y0 > y1:
                                #     y0, y1 = line["polygon"][5], line["polygon"][1]


                                #列の幅・高さ取得
                                l_width = int(abs(x1-x0))
                                l_height = int(abs(y1-y0))
                                if l_width >= l_height:
                                    vh = 'horizontal'
                                    if get_east_asian_width_count(line['content']) == 1 or len(line['content']) == 1:
                                        #縦書きメインであれば vh = 'vertical'
                                        vh = 'horizontal'
                                    else:
                                        # vh = 'horizontal'
                                        if len(line['content']) == 2 and unicodedata.east_asian_width(line['content'][0]) in 'HNa':
                                            vh = 'vertical'
                                        else:
                                            vh = 'horizontal'
                                else:
                                    # vh = 'vertical'
                                    if get_east_asian_width_count(line['content']) == 2:
                                        vh = 'vertical'
                                    elif line['content'] == 1: #横書きメインの1文字
                                        vh = 'horizontal'
                                    else:
                                        vh = 'vertical'

                                plist.append([offset, offset+length, int(x0), int(y0), int(x1), int(y1), 0, 0, 0, cl, vh])

                                #1文字修正テスト
                                for l, x in enumerate(plist):
                                    if l<=2:
                                        continue
                                    else:
                                        if plist[l-1][1]-plist[l-1][0]==1 and plist[l-2][10]=='horizontal' and plist[l-1][10]=='vertical' and plist[l][10]=='horizontal':
                                            plist[l-1][10]='horizontal'
                                        elif plist[l-1][1]-plist[l-1][0]==1 and plist[l-2][10]=='vertical' and plist[l-1][10]=='horizontal' and plist[l][10]=='vertical':
                                            plist[l-1][10]='vertical'

                            #plistの文字数分割
                            plist2 = []
                            for p in plist:
                                l = 1
                                for i in range(p[0] , p[1]+1, 1):
                                    plist2.append([l, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]])
                                    l=l+1

                            for l, word in enumerate(json_load['pages'][0]['words']):
                                cw = word["span"]["offset"]
                                cc = word['confidence']
                                t = word["content"].replace("~", "～").replace("·", "・")

                                try:
                                    x0, y0, x1, y1 = word["polygon"][0], word["polygon"][1], word["polygon"][4], word["polygon"][5]
                                except KeyError:
                                    x0, y0, x1, y1 = word["polygon"][0], 0, word["polygon"][4], 0

                                vh, rotate, mx, my, w, h = "", "", 0, 0, 0, 0
                                if x0<x1 and y0<y1: #文字回転000
                                    x0, y0, x1, y1 = word["polygon"][0], word["polygon"][1], word["polygon"][4], word["polygon"][5]
                                    rotate = 'r000'
                                elif x0>x1 and y0<y1: #文字回転090
                                    x0, y0, x1, y1 = word["polygon"][6], word["polygon"][7], word["polygon"][2], word["polygon"][3]
                                    rotate = 'r090'
                                elif x0>x1 and y0>y1: #文字回転180
                                    x0, y0, x1, y1 = word["polygon"][4], word["polygon"][5], word["polygon"][0], word["polygon"][1]
                                    rotate = 'r180'
                                elif x0<x1 and y0>y1: #文字回転270
                                    x0, y0, x1, y1 = word["polygon"][2], word["polygon"][7], word["polygon"][6], word["polygon"][3]
                                    rotate = 'r270'
                                mx, my, w, h = int((x0+x1)*0.5), int((y0+y1)*0.5), int(abs(x1-x0)), int(abs(y1-y0))
                                alist.append([t, plist2[cw][0],  0, int(x0), int(y0), int(x1), int(y1), mx, my, w, h, cc, rotate, plist2[cw][10],
                                            plist2[cw][2], plist2[cw][3], plist2[cw][4], plist2[cw][5], plist2[cw][6], plist2[cw][7], plist2[cw][8], plist2[cw][9], ceil(plist2[cw][2], 20), 0, 0])

                                if l>1:
                                    alist[-1][23] = alist[-1][3]-alist[-2][5]
                                    alist[-1][24] = alist[-1][7]-alist[-2][7]
                                continue

                        alist = del_chara(alist)
                        alist = half_width_division(alist)

                        alist = sorted(alist, key=lambda x:(x[21], x[1], x[2]))
                        with open(dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '_alist.csv', mode='w', encoding='utf-8') as f:
                            writer = csv.writer(f, lineterminator='\n', delimiter='\t')
                            writer.writerows(alist)

                        ### 市民タイムス行分割処理ここから ###
                        #行分割対象
                        o=0
                        for l in range(0, len(alist), 1):
                            if len(alist) <= l+1:
                                break

                            if alist[l][1]==1:
                                o=0

                            if alist[l][2]>0:
                                o=o+1
                                continue
                            
                            # if l == 1948:
                            #     print('NG')

                            if  (alist[l][1]==1 or 6<=alist[l][1]<=15+o) and alist[l][9]<=55 and alist[l-1][13] == 'horizontal'and alist[l][13] == 'horizontal':
                                if alist[l][1]==1:
                                    before=alist[l-1][1]
                                    o = 0

                                if (10<=alist[l][1]+o<=15+o and alist[l][0] in "-–ー「]|」") or (alist[l][1]==1 and alist[l][0] in "-–ー「|]」" and alist[l][11]<=0.98) or alist[l][0]=="|":
                                    print('chara_del'+ str(l))
                                    m, lx, ly = alist[l][1], alist[l][14], alist[l][15]
                                    alist.pop(l)
                                    x0, y0 = alist[l][3], alist[l][4]
                                    alist[l][1], alist[l][14], alist[l][15] = alist[l][1]-m, x0, y0

                                    if alist[l][1]>1:
                                        m = m+1
                                        alist[l][1]=1
                                    elif alist[l][1]-m<0:
                                        alist[l][1]=1
                                    for n in range(l+1, len(alist), 1):
                                        if alist[n][14] == lx and alist[n][15] == ly:
                                            alist[n][1], alist[n][14], alist[n][15] = alist[n][1]-m, x0, y0
                                        else:
                                            l=l-1
                                            o=0
                                            break

                                if (12<=alist[l][1]+o<=15+o and alist[l-1][0] in "、。)・一」"):
                                    print('chara_skip'+ str(l))
                                    m, lx, ly = alist[l][1], alist[l][14], alist[l][15]
                                    x0, y0 = alist[l][3], alist[l][4]
                                    alist[l][1], alist[l][14], alist[l][15] = alist[l][1]-m, x0, y0

                                    if alist[l][1]>1:
                                        m = m+1
                                        alist[l][1]=1
                                    elif alist[l][1]-m<0:
                                        alist[l][1]=1
                                    for n in range(l+1, len(alist), 1):
                                        if alist[n][14] == lx and alist[n][15] == ly:
                                            alist[n][1], alist[n][14], alist[n][15] = alist[n][1]-m, x0, y0
                                        else:
                                            l=l-1
                                            o=0
                                            break

                                if (12<=alist[l][1]+o<=15+o and alist[l][11]<=0.7):
                                    print('confidence'+ str(l))
                                    m, lx, ly = alist[l][1], alist[l][14], alist[l][15]
                                    x0, y0 = alist[l][3], alist[l][4]
                                    alist[l][1], alist[l][14], alist[l][15] = alist[l][1]-m, x0, y0

                                    if alist[l][1]>1:
                                        m = m+1
                                        alist[l][1]=1
                                    elif alist[l][1]-m<0:
                                        alist[l][1]=1
                                    for n in range(l+1, len(alist), 1):
                                        if alist[n][14] == lx and alist[n][15] == ly:
                                            alist[n][1], alist[n][14], alist[n][15] = alist[n][1]-m, x0, y0
                                        else:
                                            l=l-1
                                            o=0
                                            break

                                #スペース
                                elif (abs(alist[l-1][1]+alist[l-1][2]-alist[l][1])>=2 and alist[l][1]!=1 and not alist[l][0] in '、()（）「」"“…・' and not alist[l-1][0] in '、()（）「」"“…・') or int(alist[l][24])>=55:
                                    print('space'+ str(l))
                                    m, x0, y0, lx, ly = alist[l-1][1], alist[l][3], alist[l][4], alist[l-1][14], alist[l-1][15]
                                    for n in range(l, len(alist), 1):
                                        if alist[n][14] == lx and alist[n][15] == ly:
                                            alist[n][1], alist[n][14], alist[n][15] = n-l+1, x0, y0
                                        else:
                                            l=l-1
                                            o=0
                                            break

                                # #スペース
                                # elif o>0 and 6<=alist[l][1]+o and abs(alist[l-1][1]+alist[l-1][2]-alist[l][1])>=2 and alist[l][1]!=1 and not alist[l][0] in '、()（）「」': # not alist[l-1][0] in '()' and not alist[l][0] in '()'スペースで分けられている行分割
                                #     print('space'+ str(l))
                                #     m, x0, y0, lx, ly = alist[l-1][1], alist[l][3], alist[l][4], alist[l-1][14], alist[l-1][15]
                                #     for n in range(l, len(alist), 1):
                                #         if alist[n][14] == lx and alist[n][15] == ly:
                                #             alist[n][1], alist[n][14], alist[n][15] = alist[n][1]-m-1, x0, y0
                                #         else:
                                #             l=l-1
                                #             o=0
                                #             break

                                #スペース2
                                # elif  6+o<=alist[l][1] and alist[l][24]>=alist[l][10]:
                                #     print('space2'+ str(l))
                                #     m, x0, y0, lx, ly = alist[l][1], alist[l][3], alist[l][4], alist[l-1][14], alist[l-1][15]
                                #     for n in range(l, len(alist), 1):
                                #         if alist[n][14] == lx and alist[n][15] == ly:
                                #             alist[n][1], alist[n][14], alist[n][15] = alist[n][1]-m+1, x0, y0
                                #         else:
                                #             l=l-1
                                #             o=0
                                #             break

                        x0 = 0
                        y0 = 0
                        for l, x in enumerate(alist):
                            if l==0 or (alist[l][1]==1 and alist[l][2]<=1):
                                x0, y0 = x[3], x[4]
                                alist[l][14], alist[l][15] = x0, y0
                            else:
                                alist[l][14], alist[l][15] = x0, y0

                        alist2 = list(reversed(alist))
                        x0 = 0
                        y0 = 0
                        for l, x in enumerate(alist2):

                            if l==0 or (alist2[l-1][1]==1 and alist2[l-1][2]<=1):
                                if x[6]<x[4]:
                                    x0, y0 = x[5], x[6]
                                else:
                                    x0, y0 = x[5], x[4]
                                alist2[l][16], alist2[l][17] = x0, y0
                            else:
                                alist2[l][16], alist2[l][17] = x0, y0

                        alist = list(reversed(alist2))

                        for l in range(0, len(alist), 1):
                            alist[l][22] = ceil(alist[l][14], 60)


                        with open(dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '_blist.csv', mode='w', encoding='utf-8') as f:
                            writer = csv.writer(f, lineterminator='\n', delimiter='\t')
                            writer.writerows(alist)

                        ### 市民タイムス行分割処理ここまで ###

                        ### 段組分割ありの場合はここから ###
                        alist = sort_vlist1(alist)
                        ### 段組分割ありの場合はここまで ###

                        ### 段組分割なしの場合はここから ###
                        #azureの認識順にする際はこれもコメントアウトする
                        # hlist = []
                        # vlist = []
                        # for x in alist:
                        #     if x[13] == 'vertical':
                        #         vlist.append(x)
                        #     elif x[13] == 'horizontal':
                        #         hlist.append(x)

                        # vlist = sorted(vlist, key=lambda x:(-x[21], x[1], x[2])) #-x[14], x[15]
                        # hlist = sorted(hlist, key=lambda x:(x[21], x[1], x[2]))
                        # alist = hlist+vlist
                        ### 段組分割なしの場合はここまで ###

                xlist = []
                xlist = alist

                with open(dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '_xlist.csv', mode='w', encoding='utf-8') as f:
                    writer = csv.writer(f, lineterminator='\n', delimiter='\t')
                    writer.writerows(xlist)

                # 1列目から15列目までを抽出
                # blist = []
                # for matrix in xlist:
                #     blist.append(matrix[:14] + [round_to_nearest_multiple(int(matrix[14]), 3)] + [round_to_nearest_multiple(int(matrix[15]), 3)])

                # with open(dname + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '.csv', mode='w', encoding='utf-8') as f:
                #     writer = csv.writer(f, lineterminator='\n', delimiter='\t')
                #     writer.writerows(blist)

                # ### 市民タイムス テキスト抽出###
                # if code == 'cp932':
                #     with open(dname.replace('03_work', '04_text') + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '.txt', mode='w', encoding='cp932', errors='backslashreplace') as f:
                #         text = ""
                #         for l, x in enumerate(xlist):
                #             text = str(xlist[l][0]).encode('cp932', errors='backslashreplace').decode('cp932')
                #             if ('\\u' in text) or ('\\U' in text) or ('\\x' in text) or (len(text)==1 and text == '\\'):
                #                 print(text)
                #                 if len(text) == 1 and text == '\\': #エスケープ文字の対応
                #                     text = '￥'
                #                     xlist[l][0] = '￥' #pdfもcp932で挿入
                #                     # continue
                #                     # text = text.replace("\u005c", "\u00a5")
                #                 else:
                #                     text = '〓'
                #                     xlist[l][0] = '〓' #pdfもcp932で挿入
                #             #print(text)
                #             if l==0 or (l>0 and xlist[l][13] == 'vertical' and xlist[l-1][13] == 'vertical' and xlist[l-1][8] <= xlist[l][8])  or (l>0 and xlist[l][13] == 'horizontal' and xlist[l-1][13] == 'horizontal' and xlist[l-1][7] <= xlist[l][7]):
                #                 f.write(text)
                #             else:
                #                 f.write('\n' + text)
                # else:
                # テキスト保存
                # with open(dname.replace('03_work', '04_text') + '\\' + os.path.splitext(os.path.basename(ifile))[0] + 'V.txt', mode='w', encoding='utf-8') as f:
                #     text = ""
                #     for l, x in enumerate(xlist):
                #         text = str(xlist[l][0])
                #         if l==0 or (l>0 and xlist[l][13] == 'vertical' and xlist[l-1][13] == 'vertical' and xlist[l-1][8] <= xlist[l][8])  or (l>0 and xlist[l][13] == 'horizontal' and xlist[l-1][13] == 'horizontal' and xlist[l-1][7] <= xlist[l][7]):
                #             f.write(text)
                #         else:
                #             f.write('\n<' + str(xlist[l][22]) + '>' + text)

                #     text = ''.join([r[0] for r in xlist])

                #     # JanomeのTokenizerを初期化
                #     tokenizer = Tokenizer()

                #     # 形態素解析を実行
                #     tokens = tokenizer.tokenize(text)

                #     # 名詞を抽出
                #     nouns = [token.base_form for token in tokens if token.part_of_speech.startswith('名詞')]
                #     nouns2 = []
                #     p = re.compile('[^あ-ん]+')
                #     for x in nouns:
                #         if len(x) >=2 and unicodedata.east_asian_width(x[0]) in 'FWA' and p.fullmatch(x):
                #             if not (x == '松本' or x == '市民' or x == 'タイムス' or x == '丁目' or x == '市内'):
                #                 nouns2.append(x)

                #     # キーワードの頻度をカウント
                #     counter = Counter(nouns2)

                #     # 上位10個のキーワードを抽出
                #     top_keywords = counter.most_common(20)

                #     # 結果を表示
                #     with open(dname.replace('03_work', '04_text') + '\\' + os.path.splitext(os.path.basename(ifile))[0] + '_keyword20.txt', mode='w', encoding='utf-8') as f4:
                #         for keyword, freq in top_keywords:
                #             f4.write('\t' + f"{keyword}: {freq}")
                #             #print(f"{keyword}: {freq}")

                #     # f3.close()

                #背景画像すり替えあり
                back_image = ifile.replace('01_input','06_back_image').replace('.jpg','.pdf').replace('.png','.pdf').replace('.tiff','.pdf').replace('.tif','.pdf').replace('.jpeg','.pdf')
                if os.path.exists(back_image):
                    if '.pdf' in back_image:
                        doc = pymupdf.open(back_image)
                        doc[0].remove_rotation()
                        page = doc[0]

                fontname = ""
                fontfile = ""
                cmyk = ()
                rgb = ()
                stroke_opacity = ""
                fill_opacity = ""
                #文字色
                if output == "TLT（透明）":
                    cmyk = (0, 0, 0, 0)
                    rgb = (1, 1, 1)
                    stroke_opacity = 0
                    fill_opacity = 0
                    # cmyk = None
                    # rgb = None
                    fontname='japan-s'
                    fontfile=''
                # elif output == "RED（赤字）":
                #     cmyk = (0, 1, 1, 0)
                #     rgb = (1, 0, 0)
                #     fontname='msgothic'
                #     fontfile = 'C:\\Windows\\Fonts\\msgothic.ttc'
                #     stroke_opacity = 1
                #     fill_opacity = 1
                # elif output == "BLACK（黒字）":
                #     cmyk = (1, 1, 1, 1)
                #     rgb = (0, 0, 0)
                #     fontfile = 'C:\\Windows\\Fonts\\msgothic.ttc'
                #     fontname='msgothic'
                #     stroke_opacity = 1
                #     fill_opacity = 1

                #並び替え逆順のままだとwlist2、昇順はwlist1
                rect = pymupdf.Rect(page.rect)
                #250807なぜか逆順
                xlist.reverse()
                if not xlist is None:
                    for xx in xlist:
                        # x = xx[3]
                        # y = xx[4]
                        #フォントサイズの倍数が小さいと半角スペースが入ってしまう
                        if xx[13] == "horizontal"  and xx[12] == 'r000': #横書
                            w, h, x, y = abs(xx[9]), abs(xx[10]), xx[3]-5, xx[15]
                            fontsize = float(max(w, h))/_dpi*72*1.0 #0.8係数は次行と重複防止
                            if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                fontsize = fontsize*1.0
                            pos = pymupdf.Point(round(float(x)/_dpi*72, 2), round(float(y)/_dpi*72+fontsize, 2))
                            if  mode == 'RGBA':
                                page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                            else:
                                page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "vertical" and xx[12] == 'r000': #縦書
                            w, h, x, y = xx[9], xx[10], xx[14], xx[4]-5
                            if output == "TLT（透明）":
                                fontsize = float(max(w, h))/_dpi*72*1.0
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72, 2), round(float(y)/_dpi*72, 2))
                                if mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=270, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=270, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "horizontal" and xx[12] == 'r090': #縦書
                            w, h, x, y = xx[9] , xx[10], xx[3]-5, xx[15]
                            if output == "TLT（透明）":
                                fontsize = float(h)/_dpi*72*1.0
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72, 2), round(float(y)/_dpi*72+fontsize*1.0, 2))
                                if  mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "vertical" and xx[12] == 'r090': #縦書
                            w, h, x, y = xx[9], xx[10], xx[14], xx[4]-5
                            if output == "TLT（透明）":
                                fontsize = float(max(w, h))/_dpi*72*1.0
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72+fontsize*1.0, 2)-fontsize*1.0, round(float(y)/_dpi*72, 2))
                                if mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=270, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=270, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "vertical" and xx[12] == 'r180': #横書
                            w, h, x, y = xx[10] , xx[9], xx[14], xx[4]
                            if output == "TLT（透明）":
                                fontsize = float(h)/_dpi*72*1.0
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72, 2), round(float(y)/_dpi*72+fontsize*1.0, 2))
                                if  mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "horizontal" and xx[12] == 'r180': #横書
                            w, h, x, y = xx[9] , xx[10], xx[3]-5, xx[15]
                            if output == "TLT（透明）":
                                fontsize = float(h)/_dpi*72*1.0
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72+fontsize*0.5, 2), round(float(y)/_dpi*72+fontsize*1.0, 2))
                                if  mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "vertical" and xx[12] == 'r270': #文字270回転
                            w, h, x, y = xx[9] , xx[10], xx[14], xx[4]
                            fontsize = float(max(w, h))/_dpi*72*1.0
                            if output == "TLT（透明）":
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72+fontsize, 2), round(float(y)/_dpi*72, 2))
                                if  mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=90, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=90, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                        elif xx[13] == "horizontal" and xx[12] == 'r270':
                            w, h, x, y = abs(xx[9]), abs(xx[10]), xx[3]-5, xx[15]
                            fontsize = float(max(w, h))/_dpi*72*1.0 #0.8係数は次行と重複防止
                            if output == "TLT（透明）":
                                if unicodedata.east_asian_width(xx[0][0:1]) in 'HNa':
                                    fontsize = fontsize*1.0
                                pos = pymupdf.Point(round(float(x)/_dpi*72, 2), round(float(y)/_dpi*72, 2))
                                if  mode == 'RGBA':
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                                else:
                                    page.insert_text(point=pos, text=xx[0], fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                else:
                    fontsize = 1
                    pos = pymupdf.Point(0, 0)
                    if  mode == 'RGBA':
                        page.insert_text(point=pos, text="", fontname=fontname, fontfile=fontfile, color=cmyk, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)
                    else:
                        page.insert_text(point=pos, text="", fontname=fontname, fontfile=fontfile, color=rgb, fontsize=fontsize, rotate=0, stroke_opacity=stroke_opacity, fill_opacity=fill_opacity, overlay=False)

                #市民タイムスはOCR画像を寝かしているため、90度回転
                page.set_rotation(90)

                #↓TLT字の背景画像挿入位置
                if output == "TLT（透明）":
                    # back_image = ifile.replace('01_input','06_back_image')
                    # if os.path.exists(back_image):
                    #     if '.pdf' in back_image:
                    #         doc2 = pymupdf.open(back_image)
                    #         #背景画像すり替えあり
                    #         page.show_pdf_page(rect, doc2, 0, 0, True)
                    #         del doc2
                    #     else:
                    #         page.insert_image(rect=rect, alpha=-1, filename=back_image, rotate=0, overlay=False)
                    # else:
                        #背景画像すり替えなし
                        # page.show_pdf_page(rect=rect, doc=doc, pno=0, keep_proportion=True, rotate=0, overlay=False)
                    if plan == "A シングル":
                        file_path = os.path.splitext(os.path.basename(ifile))[0]+'_TLT(SA).pdf'
                    elif plan == "G シングル":
                        file_path = os.path.splitext(os.path.basename(ifile))[0]+'_TLT(SG).pdf'
                    elif plan == "D ダブル":
                        file_path = os.path.splitext(os.path.basename(ifile))[0]+'_TLT(D).pdf'

                elif output == "RED（赤字）":
                    # back_image = ifile.replace('01_input','06_back_image')
                    # if os.path.exists(back_image):
                    #     if '.pdf' in back_image:
                    #         doc2 = pymupdf.open(back_image)
                    #         #背景画像すり替えあり
                    #         page.show_pdf_page(rect, doc2, 0, 0, True)
                    #         del doc2
                    #     else:
                    #         page.insert_image(rect=rect, alpha=-1, filename=back_image, rotate=0, overlay=False)
                    # else:
                    #     #背景画像すり替えなし
                    #     page.show_pdf_page(rect=rect, doc=doc, pno=0, rotate=0, overlay=False)
                    file_path = os.path.splitext(os.path.basename(ifile))[0]+'_RED.pdf'
                else:
                    file_path = os.path.splitext(os.path.basename(ifile))[0]+'_BLACK.pdf'

                doc.set_metadata({})      # clear all fields
                doc.set_metadata({
                                'producer': 'TLT-PDF（Tranceparent Layered Text - PDF）', 'format': 'PDF 1.7', 'encryption': None, 'author': 'none',
                                'modDate': 'none', 'keywords': 'none', 'title': 'none', 'creationDate': 'none',
                                'creator': 'none', 'subject': 'none'
                                })

                doc.subset_fonts()
                pymupdf.TOOLS.set_subset_fontnames()
                doc.save(watch_directory + '\\02_output\\' + file_path, garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=0)
                # doc.save('C:\\Users\\USER\\Box\\竹澤ツール（共有）\\' + file_path, garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=0)

                #単頁TLTファイルをNASへ飛ばす
                # if not '★_TLT.pdf' in file_path:
                #     shutil.move(watch_directory + '\\02_output\\' + file_path, '\\\\landisk-f39018\\disk1\\' + file_path)

                # ディレクトリがない場合、作成する
                if not os.path.exists(watch_directory + '\\01_input\\completed'):
                    os.makedirs(watch_directory + '\\01_input\\completed')

                ss = ifile
                del ocrname, ifile, doc, page, file_path, json_open, json_load, img, back_image #, page #azure_name, f, azure_size,

                gc.collect()

                # os.remove(ss.replace('01_input','06_back_image'))
                # os.remove(ss.replace('01_input','03_work').replace('.pdf','_ocr.png').replace('.tif','_ocr.png').replace('.tiff','_ocr.png').replace('.png','_ocr.png').replace('.jpg','_ocr.png').replace('.jpeg','_ocr.png'))
                shutil.move(ss, watch_directory + '\\01_input\\completed\\' + os.path.basename(ss))

                # addlist = glob.glob(watch_directory + '\\03_work\\*.*')
                # for f in addlist:
                #     os.remove(f)

                #TLT連結
                print(os.path.splitext(os.path.basename(ss[0:-10]))[0])
                addlist = glob.glob(watch_directory + '\\02_output\\' + os.path.splitext(os.path.basename(ss[0:-10]))[0] + '*★*_TLT*.pdf')

                if len(addlist) == allpage_count and allpage_count > 1:
                    # ファイル名で昇順にソート
                    addlist.sort(key=lambda x: x[0])

                    # 結合先のPDFを新規作成
                    doc = pymupdf.open()

                    #結合元PDFを開く
                    for i, ifile in enumerate(addlist):

                        infile = pymupdf.open(ifile, filetype="pdf")

                        # プログラム6｜結合先PDFと結合元PDFのページ番号を指定
                        doc_lastPage = len(doc)
                        infile_lastPage = len(infile)
                        doc.insert_pdf(infile, from_page=0, to_page=infile_lastPage, start_at=doc_lastPage, rotate=0)

                        # プログラム7｜結合元PDFを閉じる
                        infile.close()

                    doc.set_metadata({})      # clear all fields
                    doc.set_metadata({
                                    'producer': 'TLT-PDF（Tranceparent Layered Text - PDF）', 'format': 'PDF 1.7', 'encryption': None, 'author': 'none',
                                    'modDate': 'none', 'keywords': 'none', 'title': 'none', 'creationDate': 'none',
                                    'creator': 'none', 'subject': 'none'
                                    })

                    ss = os.path.splitext(os.path.basename(ss[:-10]))[0] + '_merge.pdf'
                    doc.save(watch_directory + '\\02_output\\' + ss)

                    for f in addlist:
                        os.remove(f)
                    doc.close()
                    ss = ''
