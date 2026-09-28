"""Progressive mathematical animation, preserving the measured Korean narration.

Usage: .venv/bin/python week3/video/intuition.py preview|render|assemble|check|bundle
Dependencies: Pillow, FFmpeg/FFprobe; no network or generative media required.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'intuition'
W, H, FPS = 1920, 1080, 30
BG = '#0b0d12'
FG = '#edf0f6'
MUTED = '#929baa'
GRID = '#252c38'
BLUE = '#58c4dd'
YELLOW = '#f5d76e'
GREEN = '#83c167'
RED = '#fc6255'
PURPLE = '#b49bea'
COLORS = [BLUE, YELLOW, GREEN]
KOREAN = '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf'
MATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
TIMELINE = json.loads((ROOT / 'timeline.json').read_text())


def ease(t):
    t = max(0, min(1, t))
    return t * t * (3 - 2 * t)


def ramp(t, start, length=1):
    return ease((t - start) / length)


def mix(a, b, t):
    return a + (b - a) * t


@lru_cache(maxsize=256)
def font(size, family):
    return ImageFont.truetype(family, size)


@lru_cache(maxsize=1800)
def lettering(body, size, color, family, max_width):
    f = font(size, family)
    box = f.getbbox(body)
    if box[2] - box[0] > max_width:
        f = font(max(12, int(size * max_width / (box[2] - box[0]))), family)
        box = f.getbbox(body)
    im = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8))
    ImageDraw.Draw(im).text((4 - box[0], 4 - box[1]), body, font=f, fill=color)
    return im


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W, H), BG)
        self.d = ImageDraw.Draw(self.im)

    def text(self, body, x, y, size=32, color=FG, alpha=1, align='center', math=False, width=1700):
        if alpha <= 0:
            return
        family = KOREAN if any('\uac00' <= c <= '\ud7a3' for c in body) else (SERIF if math else MATH)
        tile = lettering(str(body), size, color, family, width)
        if alpha < 1:
            tile = tile.copy()
            tile.putalpha(tile.getchannel('A').point(lambda a: round(a * alpha)))
        left = x if align == 'left' else x - tile.width if align == 'right' else x - tile.width / 2
        self.im.paste(tile, (round(left), round(y - tile.height / 2)), tile)

    def line(self, a, b, color=FG, width=3, progress=1):
        if progress <= 0:
            return
        end = (mix(a[0], b[0], progress), mix(a[1], b[1], progress))
        self.d.line([a, end], fill=color, width=width)

    def arrow(self, a, b, color=FG, progress=1):
        if progress <= 0:
            return
        b = (mix(a[0], b[0], progress), mix(a[1], b[1], progress))
        self.line(a, b, color, 3)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        points = [b] + [(b[0] - 17 * math.cos(angle + s * .45), b[1] - 17 * math.sin(angle + s * .45)) for s in [-1, 1]]
        self.d.polygon(points, fill=color)

    def rect(self, x, y, w, h, color=BLUE, fill=None, width=3):
        if w <= 0 or h <= 0:
            return
        self.d.rectangle((x, y, x + w, y + h), outline=color, fill=fill, width=width)

    def dot(self, x, y, r=7, color=BLUE):
        self.d.ellipse((x-r, y-r, x+r, y+r), fill=color)

    def bracket(self, x, y, w, label, color=BLUE):
        self.line((x, y-12), (x, y), color, 2)
        self.line((x, y), (x+w, y), color, 2)
        self.line((x+w, y), (x+w, y-12), color, 2)
        self.text(label, x+w/2, y+32, 27, color)


@lru_cache(maxsize=128)
def spoken_time(index, word, fallback):
    meta = json.loads((ROOT / 'audio' / f'{index:02}.json').read_text())
    for b in meta['boundaries']:
        if word in b['text']:
            return .35 + b['offset'] / 1e7 / TIMELINE[index]['tempo']
    return fallback


def footer(c, s, t):
    reveal = 11 if s['kind'] == 'quiz' else 6
    if s['kind'] in ('compute2', 'bandwidth2', 'switch'):
        reveal = 9
    if s['index'] == 28:
        reveal = spoken_time(28, '첫', 11)
    a = ramp(t, reveal, .8)
    c.line((300, 788), (1620, 788), GRID, 2, a)
    c.text(s['equation'], 960, 839, 38, BLUE, a, width=1740)
    c.text(s['note'], 960, 902, 26, MUTED, ramp(t, reveal+(4 if s['kind']=='quiz' else 1)), width=1740)


def vector(c, s, t):
    concrete = s['index'] == 3
    count = 3 if concrete else 6
    step = 110 if concrete else 92
    left = 800 - step * (count-1)/2
    active = min(count-1, max(0, int((t-3)/2.2)))
    values = [[1, 2, 3], [4, 5, 6], [5, 7, 9]]
    for row, y in enumerate([300, 445, 620]):
        color = COLORS[row]
        c.text('ABC'[row], 460, y, 54, color, math=True)
        c.line((left-45, y-48), (left-45, y+48), color)
        c.line((left+step*(count-1)+45, y-48), (left+step*(count-1)+45, y+48), color)
        for k in range(count):
            x = left + k*step
            a = ramp(t, .7+row*.7+k*.13)
            if row == 2:
                a = ramp(t, 3+k*2.2) if concrete else a
            c.text(str(values[row][k]) if concrete else '4 B', x, y, 46 if concrete else 29, color, a)
            if k == active and concrete and t > 3:
                c.rect(x-39, y-49, 78, 98, color)
        if not concrete:
            c.bracket(left-45, y+57, step*(count-1)+90, 'N개 원소 · 0.4 GB', color)
    if concrete:
        c.text('+', 575, 445, 48, YELLOW)
        c.line((570, 520), (1130, 520), MUTED, 2, ramp(t, 2))
        k = active
        c.text(f'{values[0][k]} + {values[1][k]} = {values[2][k]}', 1430, 410, 55, GREEN, ramp(t, 3), math=True, width=490)
        c.text('원소 하나 → 덧셈 한 번', 1430, 490, 29, MUTED, ramp(t, 4))
        c.text('F = N FLOPs', 1430, 590, 40, FG, ramp(t, 10), math=True)
    else:
        c.text('0.4 × 3', 1470, 402, 57, FG, ramp(t, 4), math=True)
        c.text('1.2 GB', 1470, 492, 73, GREEN, ramp(t, 7))
        c.text('동시에 보관하는 공간', 1470, 572, 29, MUTED, ramp(t, 8))


def flow(c, s, t):
    idx = s['index']
    if idx == 18:
        for row, (label, color) in enumerate(zip(['A 읽기', 'B 읽기', 'C 쓰기'], COLORS)):
            y = 290 + row*145
            c.text(label, 350, y+35, 35, color)
            p = ramp(t, 1+row*2, 1.5)
            c.rect(560, y, 760*p, 70, color, color)
            c.text('4N Byte', 1460, y+35, 37, color, p)
        c.bracket(560, 720, 760, '읽기 2번 + 쓰기 1번 = 12N Byte', GREEN)
        return
    if idx == 29:
        texts = [('일의 양', 'F, Q', BLUE), ('시간으로 변환', 'F / P     Q / B', YELLOW), ('병목 확인', 'max', GREEN)]
    else:
        texts = [(s['labels'][0], 'A   B', BLUE), (s['labels'][1], '+', YELLOW), (s['labels'][2], 'C', GREEN)]
    for k, (label, symbol, color) in enumerate(texts):
        x = 400 + 560*k
        a = ramp(t, .6+k*1.1)
        c.text(label, x, 310, 34, color, a, width=460)
        c.text(symbol, x, 448, 79 if len(symbol)<7 else 48, color, a, math=True, width=470)
        c.line((x-175, 550), (x+175, 550), color, 2, a)
        if k<2:
            c.arrow((x+210, 445), (x+350, 445), MUTED, ramp(t, 2+k))
    if idx != 29:
        for k in range(8):
            p = max(0, min(1, (t-3-k*.7)/6))
            if 0<p<1:
                c.dot(mix(430, 1490, p), 630, 8, BLUE if p<.5 else GREEN)
        c.text('읽기', 670, 690, 28, BLUE, ramp(t, 3))
        c.text('쓰기', 1230, 690, 28, GREEN, ramp(t, 6))
    else:
        c.text('양 → 시간 → 병목', 960, 680, 44, FG, ramp(t, 8))


def formula(c, s, t):
    i = s['index']
    compute = i in (12, 19)
    color = YELLOW if compute else BLUE
    numerator, denominator = s['labels']
    symbolic = i in (12, 13)
    c.text('계산' if compute else '데이터 이동', 370, 310, 35, color)
    c.text('t', 400, 470, 110, color, math=True)
    c.text('=', 560, 470, 68, FG, ramp(t, .5))
    substitute = 0 if symbolic else ramp(t, 4, 1.2)
    c.text('F' if compute else 'Q', 910, 365, 86, color, ramp(t, 1)*(1-substitute), math=True)
    if not symbolic:
        c.text('10⁸' if compute else '1.2 × 10⁹', 910, 365, 60, color, substitute, math=True, width=450)
    c.line((690, 475), (1130, 475), FG, 3, ramp(t, 2))
    c.text('P' if compute else 'B', 910, 578, 86, color, ramp(t, 2.5)*(1-substitute), math=True)
    if not symbolic:
        c.text('20 × 10¹²' if compute else '10¹²', 910, 578, 60, color, substitute, math=True, width=450)
    c.text(numerator, 1450, 365, 36, color, ramp(t, 3), width=570)
    c.text(denominator, 1450, 578, 34, MUTED, ramp(t, 4), width=570)
    if symbolic:
        c.text('FLOPs ÷ (FLOPs/s) = s' if compute else 'Byte ÷ (Byte/s) = s', 960, 715, 37, GREEN, ramp(t, 6), math=True)
    else:
        result = '5 μs' if compute else '1.2 ms'
        c.text(result, 410, 666, 68, GREEN, ramp(t, 7))
        c.text('0.005 ms' if compute else '0.8 ms 읽기 + 0.4 ms 쓰기', 1330, 705, 32, GREEN, ramp(t, 9))


def time_bars(c, s, t):
    i = s['index']
    change = ramp(t, 6, 3)
    if i == 15:
        a, b, maximum = 2, 5, 5
    elif i == 25:
        a, b, maximum = 3, mix(4, 2, change), 4
    else:
        a = mix(.005, .0025, change) if i == 22 else .005
        b = mix(1.2, .6, change) if i == 23 else 1.2
        maximum = 1.2
    left, width = 520, 1130
    for k in range(5):
        x = left + width*k/4
        c.line((x, 300), (x, 640), GRID, 2)
        c.text(f'{maximum*k/4:g}', x, 688, 25, MUTED)
    c.text('ms', 1740, 688, 26, MUTED)
    for row, (value, color, label) in enumerate([(a,YELLOW,'연산'),(b,BLUE,'메모리')]):
        y = 350 + row*210
        c.text(label, 200, y+27, 36, color, align='left')
        p = ramp(t, 1+row*1.5, 1.5)
        length = width*value/maximum*p
        c.rect(left, y, length, 56, color, color, 1)
        c.text(f'{value:.4g} ms', 390, y+83, 27, color, p, align='right')
    if i in (22, 23, 25):
        target = 'P × 2' if i == 22 else 'B × 2'
        c.text(target, 1450, 242, 40, GREEN, ramp(t, 5), math=True)
        c.text('기준 조건에서 독립 비교' if i != 25 else '메모리 → 연산으로 병목 이동', 960, 739, 28, MUTED, ramp(t, 8))
    elif i == 21:
        c.text('240×', 1390, 240, 72, BLUE, ramp(t, spoken_time(i, '이백사십', 7)), math=True)
        c.text('동일 축척 · 작은 막대도 실제 비율', 970, 739, 27, MUTED, ramp(t, 4))
    if i == 25 and t>9:
        c.line((left+width*3/4, 320), (left+width*3/4, 650), YELLOW, 3)


def capacity(c, s, t):
    i=s['index']
    both_capacity = i == 24
    for side, x in enumerate([360, 1220]):
        color = BLUE if side==0 else GREEN
        label = ('16 GB' if side==0 else '32 GB') if both_capacity else ('16 GB' if side==0 else '1 TB/s')
        c.text(label, x+180, 250 if both_capacity else 285, 53, color)
        if side == 0 or both_capacity:
            h = (180 if side==0 else 360) if both_capacity else 270
            c.rect(x, 650-h, 360, h, color, width=3)
            p = ramp(t, 1+side, 2)
            for k in range(12):
                xx=x+20+(k%6)*54
                yy=626-(k//6)*38
                if k<12*p:
                    c.rect(xx, yy, 43, 26, color, color)
            c.text('1.2 GB 사용' if both_capacity else '동시에 담는 양', x+180, 713, 31, MUTED)
        else:
            c.line((x-80, 385), (x+440, 385), MUTED)
            c.line((x-80, 600), (x+440, 600), MUTED)
            for k in range(18):
                phase=((t*.21+k/18)%1)
                xx=mix(x-65,x+415,phase)
                yy=428+(k%3)*64
                c.rect(xx, yy, 22, 22, BLUE, BLUE)
            c.arrow((x-70, 670), (x+435, 670), color)
            c.text('초당 지나가는 양', x+180, 713, 31, MUTED)
    if both_capacity:
        c.text('같은 데이터', 960, 420, 30, FG, ramp(t, 4))
        c.text('같은 대역폭', 960, 478, 30, FG, ramp(t, 5))
        c.text('같은 시간', 960, 574, 43, YELLOW, ramp(t, 7))


def memory(c, s, t):
    i = s['index']
    if i == 9:
        for j, (x, color, name, size, cols) in enumerate([(270,BLUE,'DRAM',25,12),(1160,YELLOW,'SRAM',55,5)]):
            c.text(name, x+220, 270, 50, color)
            for k in range(48 if j==0 else 15):
                xx=x+(k%cols)*(size+10)
                yy=355+(k//cols)*(size+10)
                c.rect(xx,yy,size,size,color, width=2)
            c.text('높은 저장 밀도' if j==0 else '빠른 접근 · 더 큰 면적',x+220,615,34,color)
        c.text('같은 면적에 담을 수 있는 양이 다르다',960,712,33,FG,ramp(t,4))
        return
    if i == 11:
        nodes=[(420,370,'L1 Cache',BLUE),(960,370,'Register',YELLOW),(1420,620,'Shared Memory',PURPLE)]
        for x,y,label,color in nodes:
            c.rect(x-190,y-55,380,110,color)
            c.text(label,x,y,35,color)
        c.arrow((615,370),(760,370),BLUE,ramp(t,1))
        c.arrow((1130,420),(1320,555),PURPLE,ramp(t,5))
        c.arrow((1230,620),(1000,430),PURPLE,ramp(t,6))
        c.text('하드웨어가 관리',420,465,28,MUTED,ramp(t,2))
        c.text('프로그램이 읽기·쓰기·동기화',1380,720,28,PURPLE,ramp(t,5))
        c.text('선택적 작업 공간',660,650,37,FG,ramp(t,7))
        return
    xs=[250,600,950,1300,1650]
    labels=['HBM','L2','L1','Register','ALU']
    c.rect(455,300,1360,350,GRID,width=2)
    c.text('GPU 연산 다이',480,270,27,MUTED,align='left')
    c.rect(805,345,970,240,GRID,width=2)
    c.text('SM 내부',1740,325,26,MUTED,align='right')
    for k,(x,label) in enumerate(zip(xs,labels)):
        a=ramp(t,.8+k*.9)
        color=YELLOW if k==4 else BLUE
        if k==0 and i==8:
            for offset in [24,16,8]:
                c.rect(x-110-offset,410-offset,220,110,GRID)
        c.rect(x-110,410,220,110,color if a>.1 else GRID)
        c.text(label,x,465,35,color,a)
        if k<4:
            c.arrow((x+120,465),(xs[k+1]-120,465),MUTED,ramp(t,1.5+k))
    c.text('적층 DRAM',250,585,28,BLUE,ramp(t,2))
    c.text('칩 내부 저장 공간 · SRAM',1000,700,34,FG,ramp(t,5))
    if i==10:
        for k in range(8):
            yy=315+k*42
            c.line((425,yy),(425,yy+22),GREEN,2)
        c.text('Q는 이 HBM 경계에서 집계',400,737,25,GREEN,ramp(t,8))
    else:
        c.text('HBM ⊂ DRAM',1450,717,39,GREEN,ramp(t,6),math=True)


def pipeline(c, s, t):
    left=460
    step=230
    stage=ramp(t,1,10)*5
    for row,label in enumerate(['읽기','계산','쓰기']):
        y=300+row*140
        c.text(label,270,y+37,36,MUTED)
        for group,color in enumerate(COLORS):
            start=row+group
            p=max(0,min(1,stage-start))
            x=left+start*step
            c.rect(x,y,step-20,74,GRID,width=2)
            if p>0:
                c.rect(x,y,(step-20)*p,74,color,color)
                c.text(f'묶음 {group+1}',x+(step-20)/2,y+37,28,BG,alpha=p)
    c.arrow((left,730),(1690,730),MUTED)
    marker=left+stage*step
    c.line((marker,255),(marker,695),FG,2)
    c.text('시간',1750,730,28,MUTED)


def comparison(c, s, t):
    i=s['index']
    if i==2:
        c.text('1,000',510,300,83,BLUE)
        c.text('FLOPs · 해야 할 일',510,380,33,BLUE)
        for k in range(100):
            x=275+(k%20)*25
            y=450+(k//20)*29
            done = k < min(100,max(0,(t-3)*10))
            c.dot(x,y,6,GREEN if done else GRID)
        c.text('점 하나 = 덧셈 10회',510,645,27,MUTED)
        c.text('100 / s',1420,330,72,YELLOW,ramp(t,1))
        c.text('FLOPs/s · 처리 속도',1420,415,33,YELLOW,ramp(t,2))
        c.text('10 s',1420,585,86,GREEN,ramp(t,6),math=True)
        c.arrow((835,460),(1050,460),FG,ramp(t,3))
        return
    if i==7:
        c.text('저장 공간',490,280,35,BLUE)
        c.text('누적 이동량',1410,280,35,YELLOW)
        p=ramp(t,5,3)
        for x,val,color,maximum in [(270,1.2,BLUE,2.4),(1190,mix(1.2,2.4,p),YELLOW,2.4)]:
            c.rect(x,600-250*val/maximum,440,250*val/maximum,color,color)
            c.text(f'{val:.1f} GB',x+220,680,50,color)
        c.text('1회 → 2회 실행',960,735,35,FG,ramp(t,5))
        return
    if i==16:
        c.text('이상적 하한',420,290,36,BLUE)
        c.text('실제 실행시간',1400,290,36,YELLOW)
        c.rect(220,420,410,75,BLUE,BLUE)
        c.rect(1130,420,410,75,BLUE,BLUE)
        c.rect(1540,420,180*ramp(t,4),75,YELLOW,YELLOW)
        c.text('max(F/P, Q/B)',420,580,40,BLUE,math=True)
        c.text('준비 비용 · 의존성 · 활용률',1390,580,32,YELLOW,ramp(t,4))
        c.text('≤',870,455,80,FG,ramp(t,2))
        return
    for j,(x,label) in enumerate(zip([480,1440],s['labels'])):
        col=[BLUE,YELLOW][j]
        c.text(label,x,350,46,col,ramp(t,.5+j),width=750)
    if i==4:
        for k in range(32):
            c.rect(200+(k%16)*34,465+(k//16)*45,25,30,BLUE,BLUE if k<int(t*5) else None)
        c.bracket(200,602,536,'32 bit = 4 Byte',BLUE)
        c.text('a + b',1440,510,78,YELLOW,ramp(t,2),math=True)
        c.text('1 FLOP',1440,640,45,GREEN,ramp(t,4))


def facts(c,s,t):
    if s['index']==27:
        for k in range(16):
            x=245+k*94
            valid=k<13
            c.rect(x,380,76,100,BLUE if valid else GRID, '#13262d' if valid else None)
            c.text(str(k),x+38,430,30,BLUE if valid else MUTED)
            if not valid:
                c.line((x+10,390),(x+66,470),RED,3,ramp(t,4))
        c.bracket(245,550,12*94+76,'유효 원소: i < N',BLUE)
        c.text('mask = False',1580,600,32,RED,ramp(t,4))
        c.text('한 load가 여러 offsets를 처리한다',960,705,40,FG,ramp(t,2))
        return
    for k,(x,label) in enumerate(zip([390,960,1530],s['labels'])):
        c.text(f'0{k+1}',x,285,26,MUTED)
        c.text(label,x,430,52,COLORS[k],ramp(t,1+k),width=490)
        c.line((x-170,520),(x+170,520),COLORS[k],3,ramp(t,2+k))
    c.text('가상 하드웨어 · 가용 공간 충분 · 재사용 없음',960,660,34,MUTED,ramp(t,6))


def code(c,s,t):
    snippets=['a = tl.load(A + offsets, mask)', 'b = tl.load(B + offsets, mask)', 'c = a + b', 'tl.store(C + offsets, c, mask)']
    amounts=['4N Byte','4N Byte','N FLOPs','4N Byte']
    times=[1,4,7,10]
    for row,(snippet,amount,start) in enumerate(zip(snippets,amounts,times)):
        y=290+row*122
        col=YELLOW if row==2 else (GREEN if row==3 else BLUE)
        a=ramp(t,start)
        c.text(f'{row+1:02}',180,y,26,MUTED)
        c.text(snippet,250,y,39,col if a>0 else MUTED,align='left',width=1120)
        c.arrow((1320,y),(1420,y),col,a)
        c.text(amount,1630,y,38,col,a)
    c.text('이동량 Q',535,735,30,BLUE,ramp(t,11))
    c.text('연산량 F',1050,735,30,YELLOW,ramp(t,11))


def quiz(c,s,t):
    answer=spoken_time(28,'첫',11)
    for k,(x,label) in enumerate(zip([480,1440],s['labels'])):
        c.text(f'0{k+1}',x,272,28,MUTED)
        c.text(label,x,365,42,COLORS[k],width=780)
        c.text('?',x,495,80,MUTED,1-ramp(t,answer))
    c.text('0.4 GB 누락',480,515,47,RED,ramp(t,answer))
    c.text('0.8 ms로 과소 추정',480,620,34,FG,ramp(t,answer+.5))
    c.text('F × 2    Q × 2',1440,515,47,YELLOW,ramp(t,answer+3),math=True)
    c.text('10 μs    /    2.4 ms',1440,620,42,GREEN,ramp(t,answer+4))
    if t<answer:
        c.line((700,715),(1220,715),GRID,5)
        c.line((700,715),(1220,715),BLUE,5,max(0,min(1,t/answer)))


def frame(index,t):
    s=TIMELINE[index]
    c=Canvas()
    c.text('TENSOR → SILICON',100,60,23,BLUE,align='left')
    c.text(s['chapter'],1820,60,25,MUTED,align='right')
    c.text(s['title'],100,143,49,FG,ramp(t,0,.7),align='left',width=1720)
    c.line((100,205),(1820,205),GRID,2)
    kind=s['kind']
    if kind=='vector': vector(c,s,t)
    elif kind=='flow': flow(c,s,t)
    elif kind=='formula': formula(c,s,t)
    elif kind in ('bars','ratio','compute2','bandwidth2','switch'): time_bars(c,s,t)
    elif kind=='capacity': capacity(c,s,t)
    elif kind in ('memory','hierarchy','shared') or index==9: memory(c,s,t)
    elif kind=='pipeline': pipeline(c,s,t)
    elif kind=='compare': comparison(c,s,t)
    elif kind=='facts': facts(c,s,t)
    elif kind=='code': code(c,s,t)
    elif kind=='quiz': quiz(c,s,t)
    else: raise ValueError(kind)
    footer(c,s,t)
    c.text(f'{index+1:02} / 30',1820,1040,21,MUTED,align='right')
    c.line((100,1070),(1820,1070),GRID,3)
    c.line((100,1070),(1820,1070),BLUE,3,(s['start']+t)/600)
    # Subtitle-safe band is y=950..1020. Short fades soften cuts without retiming audio.
    fade=min(ramp(t,0,.25),ramp(s['duration']-t,0,.25))
    if fade<1:
        c.im=Image.blend(Image.new('RGB',(W,H),BG),c.im,fade)
    return c.im


def digest(index):
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(TIMELINE[index],ensure_ascii=False).encode())
    h.update((ROOT/'audio'/f'{index:02}.json').read_bytes())
    for path in [KOREAN,MATH,SERIF]:
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def run(args):
    subprocess.run([str(a) for a in args],check=True)


def preview():
    out=OUT/'preview'
    out.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(4*480,8*270),BG)
    for i,s in enumerate(TIMELINE):
        im=frame(i,min(14,s['duration']-2))
        im.save(out/f'{i:02}.png')
        sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),((i%4)*480,(i//4)*270))
    sheet.save(out/'contact-sheet.jpg',quality=95)
    for i in [3,14,21,25,28]:
        for t in [2,7,12,17]:
            frame(i,t).save(out/f'{i:02}-{t:02}s.png')
    print(out/'contact-sheet.jpg',flush=True)


def render(only=None):
    out=OUT/'clips'
    out.mkdir(parents=True,exist_ok=True)
    for i,s in enumerate(TIMELINE):
        if only is not None and i!=only: continue
        target=out/f'{i:02}.mp4'
        meta=target.with_suffix('.json')
        signature=digest(i)
        if target.exists() and meta.exists() and json.loads(meta.read_text())['sha256']==signature:
            print(f'Cached {i:02}',flush=True)
            continue
        part=target.with_suffix('.part.mp4')
        args=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
              '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(part)]
        proc=subprocess.Popen(args,stdin=subprocess.PIPE)
        try:
            for n in range(s['frames']):
                proc.stdin.write(frame(i,n/FPS).tobytes())
            proc.stdin.close()
            if proc.wait()!=0: raise RuntimeError(f'FFmpeg failed on scene {i}')
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        part.replace(target)
        meta.write_text(json.dumps({'sha256':signature,'frames':s['frames']}))
        print(f'Rendered {i+1}/30: {s["title"]}',flush=True)


def assemble():
    for i in range(30):
        meta=OUT/'clips'/f'{i:02}.json'
        if not meta.exists() or json.loads(meta.read_text())['sha256']!=digest(i):
            raise RuntimeError(f'Scene {i} is missing or stale; render first')
    listing=OUT/'clips'/'concat.txt'
    listing.write_text('\n'.join(f"file '{i:02}.mp4'" for i in range(30)))
    style='FontName=UnDotum,FontSize=15,PrimaryColour=&H00F6F0ED,OutlineColour=&H00120D0B,BorderStyle=1,Outline=2,Shadow=0,MarginV=23,Alignment=2'
    final=ROOT/'week3-intuition.ko.mp4'
    part=final.with_suffix('.part.mp4')
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',listing,
         '-i',ROOT/'audio'/'narration.wav','-i',ROOT/'chapters.ffmetadata',
         '-map','0:v','-map','1:a','-map_metadata','2','-map_chapters','2',
         '-vf',f"subtitles='{ROOT / 'week3.ko.srt'}':force_style='{style}'",'-af','loudnorm=I=-16:TP=-1.5:LRA=11',
         '-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',
         '-ar','48000','-metadata','title=커널의 시간: 양에서 시간으로','-metadata:s:a:0','language=kor',
         '-t','600','-movflags','+faststart',part])
    part.replace(final)
    # A short sample includes vector arithmetic, quantitative comparison and bottleneck change.
    samples=[]
    for i in [3,21,25]:
        sample=OUT/f'sample-{i:02}.mp4'
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',TIMELINE[i]['start'],'-i',final,
             '-t','15','-c:v','libx264','-threads','4','-preset','fast','-crf','19','-c:a','aac',sample])
        samples.append(sample)
    short=OUT/'sample.txt'
    short.write_text('\n'.join(f"file '{p.name}'" for p in samples))
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',short,
         '-c','copy','-movflags','+faststart',ROOT/'week3-intuition-preview.ko.mp4'])


def check():
    from produce import check as source_check
    source_check()
    final=ROOT/'week3-intuition.ko.mp4'
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-show_chapters','-of','json',str(final)]))
    video=next(s for s in data['streams'] if s['codec_type']=='video')
    assert (video['width'],video['height'],video['r_frame_rate'],int(video['nb_frames']))==(W,H,'30/1',18000)
    assert any(s['codec_type']=='audio' for s in data['streams'])
    assert abs(float(data['format']['duration'])-600)<.1
    assert len(data['chapters'])==10
    for i,s in enumerate(TIMELINE):
        assert json.loads((OUT/'clips'/f'{i:02}.json').read_text())['sha256']==digest(i)
    run(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-i',final,'-f','null','-'])
    report={'resolution':[W,H],'fps':FPS,'frames':18000,'duration':600,'scenes':30,
            'chapters':len(data['chapters']),'full_decode':'passed','source_narration':'preserved',
            'sha256':hashlib.sha256(final.read_bytes()).hexdigest()}
    (ROOT/'intuition-validation.json').write_text(json.dumps(report,indent=2))
    print('PASS: full decode, 1080p30, 18000 frames, 600 s, audio, chapters, current scene renders',flush=True)


def bundle():
    files=['intuition.py','README.md','lesson.py','produce.py','timeline.json','week3.ko.srt',
           'script.ko.md','chapters.ffmetadata','intuition-validation.json','audio/narration.wav']
    files += [f'audio/{i:02}.json' for i in range(30)]
    with zipfile.ZipFile(ROOT/'week3-intuition-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,'week3-video/'+name)
    print('Source bundle saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preview','render','assemble','check','bundle'])
    parser.add_argument('--scene',type=int,choices=range(30))
    args=parser.parse_args()
    if args.command=='render': render(args.scene)
    else: globals()[args.command]()
