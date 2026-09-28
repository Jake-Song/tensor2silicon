"""Fresh, code-drawn animation for the beginner lesson. No media inputs."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H, FPS = 1920, 1080, 30
PAPER = '#F6F2E9'
INK = '#253C43'
MUTED = '#61767A'
TEAL = '#087E86'
ORANGE = '#B94828'
BLUE = '#386FB0'
GREEN = '#3D7656'
WHITE = '#FFFEFA'
LINE = '#C9D5D2'
PALE = '#E3EEEA'
FONT = '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf'
MATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


@lru_cache(maxsize=1)
def scenes():
    return json.loads((ROOT/'timeline.json').read_text())


def ease(x):
    x = max(0., min(1., x))
    return x*x*(3-2*x)


def ramp(t, start, length=1):
    return ease((t-start)/length)


@lru_cache(maxsize=4096)
def lettering(body, size, color, width):
    path = FONT if any('\uac00' <= x <= '\ud7a3' for x in body) else MATH
    font = ImageFont.truetype(path, size)
    box = font.getbbox(body)
    if box[2]-box[0] > width:
        font = ImageFont.truetype(path, max(15, int(size*width/(box[2]-box[0]))))
        box = font.getbbox(body)
    out = Image.new('RGBA', (box[2]-box[0]+8, box[3]-box[1]+8))
    ImageDraw.Draw(out).text((4-box[0],4-box[1]), body, font=font, fill=color)
    return out


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W,H), PAPER)
        self.d = ImageDraw.Draw(self.im)

    def text(self, body, x, y, size=36, color=INK, alpha=1, width=1650, left=False):
        if alpha <= 0:
            return
        tile = lettering(str(body), size, color, width)
        if alpha < 1:
            tile = tile.copy()
            tile.putalpha(tile.getchannel('A').point(lambda a: round(a*alpha)))
        self.im.paste(tile, (round(x if left else x-tile.width/2), round(y-tile.height/2)), tile)

    def box(self, x,y,w,h,fill=WHITE,outline=LINE,r=24,stroke=2):
        if w > 0 and h > 0:
            self.d.rounded_rectangle((x,y,x+w,y+h), radius=min(r,w/2,h/2), fill=fill, outline=outline, width=stroke)

    def line(self, a,b,color=LINE,width=3):
        self.d.line([a,b], fill=color, width=width)

    def arrow(self, x,y,x2,y2,color=TEAL):
        self.line((x,y),(x2,y2),color,4)
        angle = math.atan2(y2-y,x2-x)
        self.d.polygon([(x2,y2)]+[(x2-16*math.cos(angle+s*.5),y2-16*math.sin(angle+s*.5)) for s in (-1,1)],fill=color)

    def dot(self,x,y,r=8,color=TEAL):
        self.d.ellipse((x-r,y-r,x+r,y+r),fill=color)


def crate(c,x,y,size=58,color=TEAL,label=''):
    c.box(x,y,size,size,fill=PALE,outline=color,r=8,stroke=3)
    c.line((x+size*.3,y),(x+size*.3,y+size),color,2)
    if label:
        c.text(label,x+size/2,y+size/2,23,color,width=size-8)


def warehouse(c,x,y,w=300,h=250):
    c.box(x,y,w,h,outline=TEAL,stroke=4)
    c.d.polygon([(x-15,y),(x+w/2,y-65),(x+w+15,y)],fill=PALE,outline=TEAL)
    for row in range(2):
        for col in range(3):
            crate(c,x+35+col*78,y+40+row*90,58)
    c.line((x+18,y+112),(x+w-18,y+112),TEAL,4)
    c.line((x+18,y+203),(x+w-18,y+203),TEAL,4)


def chef(c,x,y):
    c.d.ellipse((x-42,y-112,x+42,y-28),fill='#F4CFB4',outline=INK,width=3)
    c.box(x-64,y-26,128,137,fill=WHITE,outline=ORANGE,stroke=3)
    c.line((x,y-20),(x,y+108),ORANGE,2)
    c.dot(x-16,y-74,4,INK)
    c.dot(x+16,y-74,4,INK)
    c.d.arc((x-15,y-64,x+15,y-42),0,180,fill=INK,width=3)
    for dx in [-34,0,34]:
        c.d.ellipse((x+dx-33,y-166,x+dx+33,y-105),fill=WHITE,outline=INK,width=3)
    c.box(x-51,y-123,102,31,fill=WHITE,outline=INK,r=5,stroke=3)
    c.box(x-138,y+87,276,27,fill=ORANGE,outline=ORANGE,r=5)
    c.line((x-106,y+114),(x-106,y+160),INK,5)
    c.line((x+106,y+114),(x+106,y+160),INK,5)


def panel(c,x,y,w,h,title,body,color=TEAL,alpha=1):
    c.box(x,y,w,h)
    c.box(x+18,y+20,8,h-40,fill=color,outline=color,r=4)
    c.text(title,x+w/2,y+min(55,h*.28),min(31,int(h*.18)),color,width=w-70)
    c.text(body,x+w/2,y+h*.66,min(47,int(h*.28)),INK,alpha,width=w-70)


def tag(c,text,x,y,w=360,color=TEAL):
    c.box(x-w/2,y-26,w,52,fill=PALE,outline=PALE,r=24)
    c.text(text,x,y,25,color,width=w-30)


@lru_cache(maxsize=128)
def spoken(index, word, fallback):
    path = ROOT/'audio'/f'{index:02}.json'
    if path.exists():
        for b in json.loads(path.read_text())['boundaries']:
            if word in b['text']:
                return .35+b['offset']/1e7/scenes()[index]['tempo']
    return fallback


def kitchen(c,s,t):
    warehouse(c,190,355)
    chef(c,975,482)
    c.box(1420,365,300,250)
    c.text('결과',1570,420,37,GREEN)
    for j in range(3):
        c.d.ellipse((1460+j*74,505,1518+j*74,525),fill=PALE,outline=GREEN,width=3)
    c.arrow(530,490,790,490)
    c.arrow(1150,490,1375,490,GREEN)
    phase=(t%7)/7
    crate(c,550+185*phase,453,42)
    if t>5:
        c.dot(1170+165*phase,488,12,GREEN)
    c.text('재료 창고',340,686,34,TEAL)
    c.text('요리사',975,686,34,ORANGE)
    c.text('완성된 음식',1570,686,34,GREEN)
    c.text('메모리',340,741,28,MUTED,ramp(t,5))
    c.text('계산 장치',975,741,28,MUTED,ramp(t,7))
    c.text('출력 데이터',1570,741,28,MUTED,ramp(t,9))


def scope(c,s,t):
    c.box(210,275,1500,485,fill=PALE,outline=TEAL,stroke=3)
    tag(c,'GPU 안의 작은 계산 프로그램 = 커널',960,319,820)
    for j,(a,b,col) in enumerate([('입력','숫자 목록 A · B',TEAL),('계산','같은 자리끼리 +',ORANGE),('출력','새 숫자 목록 C',GREEN)]):
        panel(c,280+j*480,405,400,235,a,b,col,ramp(t,1+j*3))
        if j<2:c.arrow(690+j*480,520,750+j*480,520)
    c.text('이번에 재는 범위',960,716,32,INK)
    c.text('밖에서 입력 보내기 · 프로그램 준비하기는 제외',960,804,29,MUTED,ramp(t,9))


def vector(c,s,t):
    rows=[('A',[1,2,3],TEAL),('B',[4,5,6],BLUE),('C',[5,7,9],GREEN)]
    for row,(label,values,col) in enumerate(rows):
        y=305+row*158
        c.text(label,360,y+54,48,col)
        for j,v in enumerate(values):
            x=520+j*330
            c.box(x,y,240,106,outline=col,stroke=3)
            a=1 if row<2 else ramp(t,5+j*2)
            c.text(v,x+120,y+54,58,col,a)
            if row==1:
                c.text('+',x+120,y-30,31,ORANGE)
                c.text('=',x+120,y+128,31,ORANGE)
    for j in range(3):
        c.text('덧셈 1번',640+j*330,814,30,ORANGE,ramp(t,5+j*2))


def work(c,s,t):
    panel(c,180,300,620,215,'해야 할 일','접시 1,000개',TEAL)
    panel(c,1120,300,620,215,'처리 속도','초당 100개',ORANGE,ramp(t,2))
    c.text('÷',960,403,80,INK)
    c.box(455,590,1010,170,fill=PALE,outline=PALE)
    c.text('1,000 ÷ 100 = 10초',960,675,65,INK,ramp(t,5))
    c.text('계산 횟수: FLOPs',490,555,28,TEAL,ramp(t,8))
    c.text('초당 계산 횟수: FLOPs/s',1430,555,28,ORANGE,ramp(t,10))


def precision(c,s,t):
    tag(c,'숫자 하나를 저장하는 방식',960,293,630)
    c.text('FP32',500,442,92,TEAL)
    for j in range(4):
        c.box(850+j*185,382,150,118,outline=TEAL,stroke=3)
        c.text('1 Byte',925+j*185,441,26,TEAL,ramp(t,2+j*.7))
    c.text('32 bit = 4 Byte',1210,570,42,INK,ramp(t,4))
    c.line((230,630),(1690,630))
    c.text('숫자를 저장하는 크기',520,705,34,TEAL)
    c.text('덧셈 1번은 1 FLOP',1270,705,43,ORANGE,ramp(t,7))


def capacity(c,s,t):
    c.box(140,270,760,500)
    c.box(1020,270,760,500)
    warehouse(c,370,420,300,240)
    c.text('용량',520,320,38,TEAL)
    c.text('얼마나 담을 수 있나?',520,718,31,INK)
    c.text('대역폭',1400,320,38,ORANGE)
    c.arrow(1140,510,1660,510,ORANGE)
    for j in range(3):
        phase=((t/4+j/3)%1)
        crate(c,1150+440*phase,469,48,ORANGE)
    c.text('초당 얼마나 옮기나?',1400,718,31,INK)


def storage(c,s,t):
    for j,(name,color) in enumerate([('입력 A',TEAL),('입력 B',BLUE),('결과 C',GREEN)]):
        x=190+j*535
        panel(c,x,320,465,300,name,'0.4 GB',color,ramp(t,3+j*2))
        if j<2:c.text('+',x+502,470,48,INK,ramp(t,4+j*2))
    c.text('1억 개 × 4 Byte × 3목록',960,697,39,INK,ramp(t,5))
    c.text('필요한 빈 공간 1.2 GB',960,773,44,TEAL,ramp(t,9))


def reuse(c,s,t):
    warehouse(c,250,355)
    c.text('같은 저장 공간',400,700,31,TEAL)
    c.text('1.2 GB',400,755,40,TEAL)
    for j in range(2):
        y=350+j*205
        c.arrow(680,y+60,970,y+60,ORANGE)
        panel(c,1010,y-10,610,145,f'{j+1}번째 실행','1.2 GB 이동',ORANGE,ramp(t,2+j*5))
    c.text('누적 2.4 GB',1320,773,46,ORANGE,ramp(t,9))


def memorytypes(c,s,t):
    c.box(160,290,740,420)
    c.box(1020,290,740,420)
    c.text('큰 재료 창고',530,345,31,TEAL)
    c.text('가까운 작은 작업대',1390,345,31,ORANGE)
    c.text('HBM',530,477,72,TEAL)
    c.text('SRAM',1390,477,72,ORANGE,ramp(t,5))
    c.text('많은 데이터 보관',530,591,30,INK)
    c.text('DRAM을 층층이 쌓은 메모리',530,655,27,MUTED,ramp(t,3))
    c.text('빠른 접근에 유리',1390,591,30,INK,ramp(t,6))
    c.text('같은 용량에 더 큰 칩 면적 필요',1390,655,27,MUTED,ramp(t,9))
    c.text('서로 다른 장점을 함께 사용합니다',960,786,39,TEAL,ramp(t,10))


def hierarchy(c,s,t):
    cards=[('HBM','큰 창고',TEAL),('캐시','가까운 선반',BLUE),('레지스터','손에 든 재료',ORANGE),('계산','요리하기',GREEN)]
    for j,(a,b,col) in enumerate(cards):
        x=140+j*440
        panel(c,x,315,360,235,a,b,col,ramp(t,j*2))
        if j<3:c.arrow(x+375,430,x+424,430)
    c.box(430,635,1060,135,fill=PALE,outline=PALE)
    c.text('공유 메모리: 프로그램이 직접 관리하는 작업대',960,676,31,TEAL,ramp(t,7))
    c.text('오늘의 덧셈에서 꼭 거쳐야 하는 곳은 아닙니다',960,733,28,INK,ramp(t,10))


def formulas(c,s,t):
    panel(c,150,310,770,355,'계산 시간','계산 횟수 ÷ 계산 속도',ORANGE)
    panel(c,1000,310,770,355,'이동 시간','읽고 쓸 양 ÷ 대역폭',TEAL,ramp(t,3))
    c.text('F ÷ P',535,611,38,ORANGE,ramp(t,6))
    c.text('Q ÷ B',1385,611,38,TEAL,ramp(t,7))
    c.text('두 식 모두 “일의 양 ÷ 속도”',960,768,44,INK,ramp(t,8))


def pipeline(c,s,t):
    left=400
    for j,name in enumerate(['첫 묶음','다음 묶음','그다음 묶음']):
        y=315+j*140
        c.text(name,250,y+40,31,INK)
        for k,(word,col) in enumerate([('읽기',TEAL),('계산',ORANGE),('쓰기',GREEN)]):
            x=left+(j+k)*245
            a=ramp(t,1+(j+k)*1.3)
            c.box(x,y,220,80,fill=PALE,outline=col,stroke=3)
            c.text(word,x+110,y+40,31,col,a)
    c.arrow(left,790,1690,790,INK)
    c.text('시간 →',1730,790,25,MUTED)
    phase=(t%10)/10
    c.line((left+1280*phase,285),(left+1280*phase,735),BLUE,3)
    c.text('다른 묶음의 작업은 동시에 진행할 수 있어요',960,241,31,TEAL)


def overlap(c,s,t):
    c.text('따로 진행',300,350,35,INK)
    c.box(570,307,310,90,fill=ORANGE,outline=ORANGE)
    c.text('계산 2',725,352,31,WHITE)
    c.box(890,307,775*ramp(t,1),90,fill=TEAL,outline=TEAL)
    c.text('이동 5',1270,352,31,WHITE,ramp(t,2))
    c.text('합계 7 ms',1520,444,35,INK,ramp(t,4))
    c.text('충분히 겹치기',300,606,35,INK,width=350)
    c.box(570,540,310,65,fill=ORANGE,outline=ORANGE)
    c.text('계산 2',725,572,28,WHITE)
    c.box(570,620,775*ramp(t,5),65,fill=TEAL,outline=TEAL)
    c.text('이동 5',957,652,28,WHITE,ramp(t,6))
    c.text('큰 값 5 ms',1520,641,35,TEAL,ramp(t,7))
    c.text('1 ms = 1/1,000초',960,790,32,MUTED)


def ideal(c,s,t):
    panel(c,165,315,740,330,'이상적인 계산','최대 성능 · 충분한 중첩',TEAL)
    panel(c,1015,315,740,330,'실제 실행','준비 · 기다림 · 접근 방식',ORANGE,ramp(t,3))
    c.text('더 빨라지기 어려운 기준',535,711,32,TEAL)
    c.text('직접 측정해서 확인',1385,711,36,ORANGE,ramp(t,7))


def assumptions(c,s,t):
    cards=[('숫자 목록 길이','N = 1억',TEAL),('숫자 한 개의 크기','FP32 = 4 Byte',BLUE),('계산 속도','초당 덧셈 20조 번',ORANGE),('메모리 대역폭','초당 1,000 GB',GREEN)]
    for j,(title,body,col) in enumerate(cards):
        panel(c,180+(j%2)*825,285+(j//2)*255,735,212,title,body,col,ramp(t,j*2))
    c.text('연습용 가상 조건 · 입력 한 번씩 읽기 · 출력 한 번 쓰기',960,823,29,MUTED)


def bytes_scene(c,s,t):
    for j,(title,col) in enumerate([('A 읽기',TEAL),('B 읽기',BLUE),('C 쓰기',GREEN)]):
        panel(c,170+j*555,310,470,275,title,'0.4 GB',col,ramp(t,j*3))
        c.text('4 × 1억 Byte',405+j*555,634,29,col,ramp(t,j*3))
        if j<2:c.text('+',681+j*555,440,55,INK)
    c.text('모두 합쳐 1.2 GB 이동',960,764,52,TEAL,ramp(t,8))


def numeric(c,s,t):
    compute=s['kind']=='compute'
    col=ORANGE if compute else TEAL
    a,b=('1억 번','초당 20조 번') if compute else ('1.2 GB','초당 1,000 GB')
    result='0.000005초' if compute else '0.0012초'
    unit='5 μs = 0.005 ms' if compute else '1.2 ms'
    panel(c,180,290,635,220,'일의 양',a,col)
    panel(c,1105,290,635,220,'처리 속도',b,col,ramp(t,2))
    c.text('÷',960,401,74,INK)
    c.text('=',420,624,62,col,ramp(t,4))
    c.text(result,1030,623,71,col,ramp(t,4))
    c.box(440,714,1040,108,fill=PALE,outline=PALE)
    c.text(unit,960,768,51,col,ramp(t,7))


def timebars(c,values,labels,y=375,scale=1.2,width=1220,delay=0,t=99):
    for j,(value,label,col) in enumerate(zip(values,labels,[ORANGE,TEAL])):
        yy=y+j*166
        c.text(label,205,yy+43,30,col,width=290)
        c.box(380,yy,width,86,fill=WHITE,outline=LINE,r=8)
        extent=width*value/scale*ramp(t,delay+j*2)
        if extent>0:c.box(380,yy,extent,86,fill=col,outline=col,r=0)
        c.text(f'{value:g} ms',1723,yy+43,35,col,width=235)


def compare(c,s,t):
    timebars(c,[.005,1.2],['계산','데이터 이동'],t=t)
    c.text('같은 길이 기준으로 그린 막대',960,303,28,MUTED)
    c.text('240배',960,752,83,TEAL,ramp(t,spoken(s['index'],'이백사십',8)))


def upgrade(c,s,t):
    for j,(title,body,col) in enumerate([('원래 조건','1.2 ms',INK),('계산 성능만 2배','1.2 ms',ORANGE),('대역폭만 2배','0.6 ms',TEAL)]):
        x=165+j*565
        panel(c,x,330,465,315,title,body,col,ramp(t,j*4))
        c.text('전체 시간의 기준',x+232,698,28,MUTED,ramp(t,j*4))
    c.text('각 변경은 원래 조건에서 따로 비교합니다',960,794,31,MUTED)


def storage_only(c,s,t):
    c.box(210,310,680,405)
    c.box(1030,310,680,405)
    for x,cap,w in [(550,'16 GB',150),(1370,'32 GB',300)]:
        c.box(x-w/2,455,w,140,fill=PALE,outline=TEAL,stroke=3)
        c.text(cap,x,378,52,TEAL)
        c.text('공간 충분',x,640,32,INK)
    c.arrow(917,505,998,505)
    c.text('옮길 양과 속도가 같으면 → 시간도 1.2 ms',960,806,39,INK,ramp(t,7))


def switch(c,s,t):
    for j,(title,mem,total) in enumerate([('개선 전',4,4),('대역폭 2배',2,3)]):
        x=120+j*930
        c.box(x,285,870,483)
        c.text(title,x+435,334,35,INK)
        for k,(label,val,col) in enumerate([('계산',3,ORANGE),('이동',mem,TEAL)]):
            y=420+k*123
            c.text(label,x+105,y+32,29,col)
            c.box(x+200,y,440*val/4*ramp(t,j*6+k),62,fill=col,outline=col,r=5)
            c.text(f'{val} ms',x+746,y+32,33,col)
        c.text(f'큰 값: {total} ms',x+435,695,44,INK,ramp(t,3+j*6))


def code(c,s,t):
    lines=[('a = 입력_A[i] 읽기','A에서 꺼내기',TEAL),('b = 입력_B[i] 읽기','B에서 꺼내기',BLUE),('c = a + b','두 값 더하기',ORANGE),('결과_C[i] = c 저장','결과 넣기',GREEN)]
    c.box(155,275,1020,500,fill=INK,outline=INK)
    for j,(line,meaning,col) in enumerate(lines):
        y=350+j*113
        c.text(f'{j+1:02}',205,y,26,'#9CBABB',left=True)
        c.text(line,295,y,37,WHITE,ramp(t,j*2),width=810,left=True)
        c.text(meaning,1500,y,36,col,ramp(t,j*2),width=540)
    c.text('개념을 보여주는 의사 코드 · i는 목록 안의 자리',960,824,28,MUTED)


def quiz(c,s,t):
    output=s['kind']=='quiz_output'
    question='출력 쓰기를 빼면 무엇이 빠질까요?' if output else '목록 길이가 2배라면 일의 양은?'
    answer='출력 0.4 GB를 빠뜨립니다' if output else '계산량도, 이동량도 2배'
    detail='0.8 ms로 과소 예상 → 실제 기준 1.2 ms' if output else '계산 10 μs · 데이터 이동 2.4 ms'
    reveal=spoken(s['index'],'정답',s['duration']*.5)
    c.box(210,290,1500,235)
    c.text(question,960,407,49,INK,width=1370)
    if t<reveal:
        for j in range(3):
            c.dot(900+j*60,660,10,TEAL if (int(t*1.5)%3)==j else LINE)
        c.text('잠시 생각해 보세요',960,748,32,MUTED)
    else:
        c.text(answer,960,633,51,TEAL,ramp(t,reveal,.65))
        c.text(detail,960,748,36,INK,ramp(t,reveal+.8),width=1700)


def end(c,s,t):
    data=[('01','양을 세기','계산 횟수 · 읽고 쓸 양',TEAL),('02','시간 구하기','각각의 양 ÷ 속도',ORANGE),('03','비교하기','충분히 겹치면 큰 시간',GREEN)]
    for j,(n,title,body,col) in enumerate(data):
        x=140+j*600
        c.box(x,310,440,410)
        c.text(n,x+220,385,52,col)
        c.text(title,x+220,496,46,INK,ramp(t,j*2))
        c.text(body,x+220,641,28,col,ramp(t,1+j*2),width=405)
        if j<2:c.arrow(x+468,507,x+570,507)
    c.text('다음 이야기: 가까이 둔 데이터를 다시 써서 이동 줄이기',960,807,32,MUTED,ramp(t,9))


DRAW = dict(kitchen=kitchen,scope=scope,vector=vector,work=work,precision=precision,
            capacity=capacity,storage=storage,reuse=reuse,memorytypes=memorytypes,
            hierarchy=hierarchy,formulas=formulas,pipeline=pipeline,overlap=overlap,
            ideal=ideal,assumptions=assumptions,bytes=bytes_scene,compute=numeric,memory=numeric,
            compare=compare,upgrade=upgrade,storage_only=storage_only,switch=switch,
            code=code,quiz_output=quiz,quiz_size=quiz,end=end)


def frame(index,t):
    s=scenes()[index]
    c=Canvas()
    c.text('GPU를 처음 배우는 시간',105,62,23,TEAL,left=True)
    c.text(s['chapter'],1540,62,24,MUTED,width=540)
    c.text(s['title'],960,153,48,INK,width=1750)
    c.line((105,217),(1815,217),LINE,2)
    DRAW[s['kind']](c,s,t)
    reveal=4
    if s['kind'] in ('quiz_output','quiz_size'):
        reveal=spoken(index,'정답',s['duration']*.5)
    elif s['kind']=='compare':
        reveal=spoken(index,'이백사십',8)
    c.text(s['takeaway'],960,902,29,TEAL,ramp(t,reveal),width=1740)
    c.box(0,951,W,129,fill=INK,outline=INK,r=0)
    c.text(f'{index+1:02} / {len(scenes()):02}',1817,931,18,MUTED)
    progress=(s['start']+t)/600
    c.line((0,1077),(W,1077),MUTED,5)
    c.line((0,1077),(W*progress,1077),'#4FBBAD',5)
    fade=min(ramp(t,0,.25),ramp(s['duration']-t,0,.25))
    if fade<1:
        c.im=Image.blend(Image.new('RGB',(W,H),PAPER),c.im,fade)
    return c.im


def signature(index):
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(scenes()[index],ensure_ascii=False).encode())
    h.update((ROOT/'audio'/f'{index:02}.json').read_bytes())
    for font in (FONT,MATH):h.update(Path(font).read_bytes())
    return h.hexdigest()


def render_one(index):
    s=scenes()[index]
    out=ROOT/'clips'
    out.mkdir(exist_ok=True)
    target=out/f'{index:02}.mp4'
    meta=target.with_suffix('.json')
    sig=signature(index)
    if target.exists() and meta.exists() and json.loads(meta.read_text())['signature']==sig:
        print(f'Cached {index+1}/{len(scenes())}',flush=True)
        return
    part=target.with_suffix('.part.mp4')
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an',
        '-c:v','libx264','-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(part)],stdin=subprocess.PIPE)
    try:
        for n in range(s['frames']):proc.stdin.write(frame(index,n/FPS).tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f'Render failed: scene {index}')
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    part.replace(target)
    meta.write_text(json.dumps(dict(signature=sig,frames=s['frames'])))
    print(f'Rendered {index+1}/{len(scenes())}: {s["title"]}',flush=True)


def render(only=None):
    if only is not None:
        render_one(only)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(render_one,range(len(scenes()))))


def preview():
    out=ROOT/'preview'
    out.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(4*480,math.ceil(len(scenes())/4)*270),PAPER)
    for i,s in enumerate(scenes()):
        t=min(s['duration']-1,max(15,s['duration']*.75))
        im=frame(i,t)
        im.save(out/f'{i:02}.png')
        sheet.paste(im.resize((480,270)),((i%4)*480,(i//4)*270))
    sheet.save(out/'contact-sheet.jpg',quality=94)
    print(out/'contact-sheet.jpg',flush=True)
