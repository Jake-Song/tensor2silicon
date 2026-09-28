"""Deterministic 1080p diagrams. All matrices, labels, and motion are drawn locally."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
from facts import FACTS

ROOT=Path(__file__).resolve().parent
W,H,FPS=1920,1080,30
BG='#0B1220'; PANEL='#141F32'; EDGE='#293A52'; INK='#EDF2FA'; MUTED='#9CAFC5'
TEAL='#54D9CD'; ORANGE='#FFB366'; PURPLE='#BAA2FF'; BLUE='#70AEFF'; RED='#F18F9B'
FONT='/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf'
MATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
NODES=[('embed','Embedding'),('norm1','LayerNorm ①'),('qkv','Q · K · V     ①②③'),
       ('rope','RoPE / Head 분리'),('qk','QKᵀ                  ④'),('softmax','Scale / Mask / Softmax'),
       ('pv','PV                     ⑤'),('out','Merge / Output ⑥'),('add1','Add / LayerNorm ②'),
       ('ffn','Gate · Up          ⑦⑧'),('gate','SiLU / 원소별 곱'),('down','Down ⑨ / Add'),('head','Final norm / LM head')]


@lru_cache(None)
def scenes():
    return json.loads((ROOT/'timeline.json').read_text())


@lru_cache(256)
def font(size,korean=False):
    return ImageFont.truetype(FONT if korean else MATH,size)


@lru_cache(2048)
def label(text,size,color,width):
    korean=any('\uac00'<=c<='\ud7a3' or c in '①②③④⑤⑥⑦⑧⑨' for c in text)
    # UnDotum lacks superscript T, minus, and approximation signs. Keep math
    # glyphs in DejaVu even when the same label contains Korean/circled digits.
    runs=[]
    for char in text:
        use_korean=korean and supports_korean_font(char)
        if runs and runs[-1][0]==use_korean:
            runs[-1][1]+=char
        else:
            runs.append([use_korean,char])
    extent=sum(font(size,k).getlength(txt) for k,txt in runs)
    if extent>width:
        size=max(17,int(size*width/extent))
    extent=sum(font(size,k).getlength(txt) for k,txt in runs)
    im=Image.new('RGBA',(math.ceil(extent)+12,size*2+12))
    d=ImageDraw.Draw(im); x=5
    for k,txt in runs:
        f=font(size,k)
        d.text((x,size+5),txt,font=f,fill=color,anchor='ls')
        x+=f.getlength(txt)
    bounds=im.getbbox()
    return im.crop(bounds) if bounds else im


@lru_cache(4096)
def supports_korean_font(char):
    f=font(32,True)
    return bytes(f.getmask(char))!=bytes(f.getmask('\U0010ffff'))


def text(im,body,x,y,size=32,color=INK,width=1300,left=False):
    tile=label(str(body),size,color,width)
    im.paste(tile,(round(x if left else x-tile.width/2),round(y-tile.height/2)),tile)


def box(im,x,y,w,h,color=PANEL,outline=EDGE,r=18,stroke=2):
    if w>0 and h>0:
        ImageDraw.Draw(im).rounded_rectangle((x,y,x+w,y+h),radius=min(r,w/2,h/2),fill=color,outline=outline,width=stroke)


def line(im,a,b,color=EDGE,width=3):
    ImageDraw.Draw(im).line([a,b],fill=color,width=width)


def arrow(im,a,b,color=TEAL):
    line(im,a,b,color,3)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    pts=[b]+[(b[0]-13*math.cos(angle+s*.5),b[1]-13*math.sin(angle+s*.5)) for s in (-1,1)]
    ImageDraw.Draw(im).polygon(pts,fill=color)


def ease(t):
    t=max(0,min(1,t)); return t*t*(3-2*t)


def matrix(im,x,y,rows,cols,color=TEAL,cell=40,active_row=None,active_col=None,active_cell=None,causal=False,values=None,title='',shape=''):
    for r in range(rows):
        for c in range(cols):
            fill='#172A3C'
            if active_row==r or active_col==c or active_cell==(r,c): fill=color
            if causal:
                fill='#263C50' if c<=r else '#141A26'
            box(im,x+c*cell,y+r*cell,cell-5,cell-5,fill,color if not causal or c<=r else EDGE,5,1)
            if values is not None:
                value=str(values[r][c])
                text(im,value,x+c*cell+(cell-5)/2,y+r*cell+(cell-5)/2,19,INK if fill!=color else BG,width=cell-5)
    if title:text(im,title,x+cols*cell/2,y-32,29,color,width=cols*cell+130)
    if shape:text(im,shape,x+cols*cell/2,y+rows*cell+28,25,MUTED,width=cols*cell+160)


def card(im,x,y,w,title,body,color=TEAL,h=140):
    box(im,x,y,w,h,outline=color)
    text(im,title,x+w/2,y+38,25,MUTED,width=w-30)
    text(im,body,x+w/2,y+91,35,color,width=w-35)


def display(value):
    if value in FACTS and isinstance(FACTS[value],(int,float)):
        n=FACTS[value]
        if value.endswith('matmul'):
            return f'{n/1e12:.3f} TFLOPs' if n>1e12 else f'{n/1e9:.3f} GFLOPs'
        return f'{n:,}'
    return value


@lru_cache(40)
def background(index):
    s=scenes()[index]
    im=Image.new('RGB',(W,H),BG)
    text(im,'TENSOR → SILICON / TRANSFORMER',64,48,23,TEAL,left=True,width=900)
    text(im,s['chapter'],1818,48,24,MUTED,width=540)
    text(im,s['title'],960,125,49,width=1790)
    line(im,(60,182),(1860,182))
    box(im,52,210,325,676,r=20)
    text(im,'전체 연산 지도',214,244,26,INK)
    for j,(key,title) in enumerate(NODES):
        yy=282+j*42
        selected=s['focus']=='all' or s['focus']==key
        if selected:box(im,68,yy-15,293,33,'#203B49',TEAL,8,1)
        text(im,title,84,yy+1,21,INK if selected else MUTED,width=260,left=True)
        if j<12:line(im,(214,yy+19),(214,yy+25),TEAL if selected else EDGE,2)
    text(im,'층 안의 matmul',150,865,19,MUTED,width=195)
    box(im,413,210,1450,506,r=24)
    if s['formula']:
        text(im,s['formula'],1138,679,33,TEAL,width=1350)
    else:
        text(im,'학습 가중치',690,680,24,ORANGE,width=240)
        text(im,'입력 activation',1130,680,24,TEAL,width=300)
        text(im,'계산 결과',1570,680,24,PURPLE,width=240)
    card(im,413,746,705,'PARAMS · 학습된 숫자의 개수',display(s['params']) or '연산별로 구분해서 세기',ORANGE,h=132)
    card(im,1138,746,725,'FLOPs · 이번 입력의 계산량',display(s['flops']) or '입력 shape와 함께 계산',PURPLE,h=132)
    text(im,s['takeaway'],960,922,29,TEAL,width=1790)
    box(im,0,964,W,116,'#070C15','#070C15',0)
    text(im,f'{index+1:02} / {len(scenes()):02}',1800,945,18,MUTED)
    return im


def listlines(im,lines,t,s,y=308,step=95,x=1138,width=1260):
    # Three reveals aligned to positions in the spoken-word stream.
    for j,body in enumerate(lines):
        reveal=s['reveal'][min(j,2)]
        if t>=reveal:
            box(im,470,y+j*step-30,1332,68,'#1B2A40',EDGE,12,1)
            text(im,body,x,y+j*step+3,34,INK if j else TEAL,width=width)


def flow(im,labels,t,y=450):
    n=len(labels); w=min(246,1210/n-35); gap=45
    total=n*w+(n-1)*gap; start=1138-total/2
    for j,body in enumerate(labels):
        x=start+j*(w+gap)
        box(im,x,y-65,w,130,outline=[TEAL,ORANGE,PURPLE][j%3])
        text(im,body,x+w/2,y,30,[TEAL,ORANGE,PURPLE][j%3],width=w-24)
        if j<n-1:
            arrow(im,(x+w+5,y),(x+w+gap-5,y))
            xx=x+w+7+(gap-14)*((t*.5)%1)
            ImageDraw.Draw(im).ellipse((xx-4,y-4,xx+4,y+4),fill=TEAL)


def draw(im,s,t):
    kind=s['kind']; p=min(1,t/max(1,s['duration']-.5))
    stage=sum(t>=r for r in s['reveal'])
    if kind=='overview':
        flow(im,['토큰 ID','Embedding','8개 층','LM head','Logits'],t)
        text(im,'입력 문장 → 토큰 벡터 → 다음 토큰의 점수',1138,290,34,INK)
        text(im,'전체 모델에서 한 층으로, 다시 개별 연산으로',1138,590,29,MUTED)
    elif kind=='block':
        flow(im,['LayerNorm','Attention','Add','LN + FFN','Add'],t)
        text(im,'서로 다른 토큰을 섞기',860,315,33,TEAL)
        text(im,'토큰별 특징을 변환하기',1445,315,33,PURPLE)
        line(im,(505,535),(1740,535),MUTED,2)
        text(im,'입력을 보존하는 residual 경로',1138,584,27,MUTED)
    elif kind in ('params','matmul','projection','down'):
        x1,x2,x3=555,1000,1490
        row=int(t*.55)%3; col=int(t*.45)%3
        shapes=('[BT,D]','[D,D]','[BT,D]') if kind=='projection' else ('[BT,F]','[F,D]','[BT,D]') if kind=='down' else ('[m,k]','[k,n]','[m,n]')
        a=[[1,2,0,1],[0,1,1,2],[2,0,1,1]]
        b=[[1,0,2],[2,1,0],[0,2,1],[1,1,1]]
        result=[[sum(a[r][k]*b[k][c] for k in range(4)) for c in range(3)] for r in range(3)]
        matrix(im,x1,353,3,4,TEAL,46,active_row=row,title='입력 X',shape=shapes[0],values=a if kind=='matmul' else None)
        matrix(im,x2,353,4,3,ORANGE,46,active_col=col,title='가중치 W',shape=shapes[1],values=b if kind=='matmul' else None)
        matrix(im,x3,353,3,3,PURPLE,46,active_cell=(row,col),title='출력 Y',shape=shapes[2],values=result if kind=='matmul' else None)
        text(im,'@',852,424,62,TEAL); text(im,'=',1350,424,62,PURPLE)
        message={'params':'W의 칸을 세면 Params / X와 Y는 activation',
                 'matmul':'행과 열의 짝 → 곱하기 → 더하기 → 출력 한 칸',
                 'projection':'Q: 각 토큰이 찾고 싶은 정보의 표현',
                 'down':'F → D로 축소한 뒤 입력 X₁을 다시 더하기'}[kind]
        if kind=='matmul':
            message=' + '.join(f'{a[row][k]}×{b[k][col]}' for k in range(4))+f' = {result[row][col]}'
        text(im,message,1138,602,32,INK,width=1280)
    elif kind=='embedding':
        matrix(im,850,314,6,5,ORANGE,43,active_row=int(t*.45)%6,title='Embedding table',shape='[V,D]')
        text(im,'토큰 ID',615,365,33,TEAL)
        text(im,str(int(t*.45)%6),615,445,69,TEAL)
        arrow(im,(710,446),(824,446))
        matrix(im,1430,410,1,5,PURPLE,43,active_row=0,title='선택한 행',shape='[D]')
        arrow(im,(1090,446),(1385,446))
        text(im,'표 전체를 곱하지 않고 필요한 행만 읽기',1138,613,32)
    elif kind=='norm':
        flow(im,['평균·분산','중심 맞추기','크기 정규화','γ × 값 + β'],t,y=422)
        text(im,'한 토큰의 특징 D개 안에서 계산',1138,294,34,TEAL)
        text(im,'γ [D] + β [D] → 파라미터 2D',1138,586,34,ORANGE)
    elif kind=='qkv':
        card(im,480,371,240,'입력','LN(X)',TEAL,h=155)
        for j,(name,col) in enumerate([('Q',PURPLE),('K',BLUE),('V',TEAL)]):
            yy=290+j*124
            arrow(im,(739,448),(975,yy+38),col)
            box(im,1000,yy,300,78,outline=ORANGE)
            text(im,'@ W'+name.lower(),1150,yy+39,34,ORANGE)
            arrow(im,(1320,yy+39),(1415,yy+39),col)
            box(im,1440,yy,300,78,outline=col)
            text(im,name+' [BT,D]',1590,yy+39,32,col)
    elif kind=='heads':
        matrix(im,505,385,3,8,TEAL,33,title='원래 벡터',shape='[B,T,D]')
        arrow(im,(803,433),(940,433))
        for j in range(4):
            matrix(im,982+j*94,351,3,2,[TEAL,BLUE,PURPLE,TEAL][j],30,title=f'h{j+1}')
        cx,cy=1610,433
        d=ImageDraw.Draw(im); d.ellipse((cx-70,cy-70,cx+70,cy+70),outline=EDGE,width=3)
        angle=t*.22
        arrow(im,(cx,cy),(cx+67*math.cos(angle),cy+67*math.sin(angle)),PURPLE)
        text(im,'RoPE',cx,301,31,PURPLE)
        text(im,'성분 쌍 회전',cx,564,29,MUTED)
        text(im,'head마다 폭 d / h × d = D',1000,607,32,TEAL)
    elif kind in ('scores','softmax','prefill','decode'):
        if kind=='decode':
            matrix(im,565,401,1,4,TEAL,46,active_row=0,title='새 Q',shape='[1,d]')
            matrix(im,1030,335,4,6,BLUE,38,active_col=int(t*.6)%6,title='K cacheᵀ',shape='[d,S]')
            matrix(im,1500,401,1,6,PURPLE,34,active_col=int(t*.6)%6,title='점수',shape='[1,S]')
            text(im,'@',872,434,55); text(im,'=',1410,434,55)
            text(im,'이전 K·V 재사용 / 새 K·V만 cache에 추가',1138,592,31)
        elif kind=='prefill':
            matrix(im,612,345,4,4,TEAL,51,causal=True,title='T 토큰',shape='T × T')
            matrix(im,1233,308,8,8,PURPLE,33,causal=True,title='2T 토큰',shape='2T × 2T')
            text(im,'→',1028,433,66,MUTED)
            text(im,'점수표 크기는 4배',1138,610,38,TEAL)
        elif kind=='scores':
            matrix(im,543,361,4,3,TEAL,47,active_row=int(t*.45)%4,title='Q',shape='[T,d]')
            matrix(im,997,361,3,4,BLUE,47,active_col=int(t*.45)%4,title='Kᵀ',shape='[d,S]')
            matrix(im,1480,361,4,4,PURPLE,47,active_cell=(int(t*.45)%4,int(t*.45)%4),title='토큰 간 점수',shape='[T,S]')
            text(im,'@',840,435,57); text(im,'=',1350,435,57)
        else:
            matrix(im,590,340,4,4,BLUE,48,causal=True,title='Causal mask',shape='미래 토큰은 차단')
            arrow(im,(870,432),(1080,432))
            vals=[['1','0','0','0'],['.4','.6','0','0'],['.2','.3','.5','0'],['.1','.2','.3','.4']]
            matrix(im,1180,340,4,4,PURPLE,55,values=vals,title='Softmax → P',shape='각 행의 합 = 1')
            text(im,'지수 함수 / 합산 / 나눗셈',1138,609,31)
    elif kind=='mix':
        for j,(weight,col) in enumerate([(.1,TEAL),(.2,BLUE),(.3,ORANGE),(.4,PURPLE)]):
            yy=298+j*76
            text(im,f'{weight:.1f}',600,yy+18,30,col)
            box(im,660,yy,weight*700,36,col,col,6)
            text(im,f'× V{j+1}',1048,yy+18,28,col)
            arrow(im,(1130,yy+18),(1380,444),col)
        card(im,1410,370,300,'가중합','Context',PURPLE,h=145)
        text(im,'관련성이 큰 토큰의 내용을 더 많이 가져오기',1138,615,31)
    elif kind=='merge':
        for j,col in enumerate((TEAL,BLUE,PURPLE)):
            matrix(im,520+j*100,380,3,2,col,30,title=f'h{j+1}')
        arrow(im,(830,430),(940,430))
        matrix(im,980,380,3,6,TEAL,30,title='merge',shape='[BT,D]')
        arrow(im,(1190,430),(1290,430))
        card(im,1310,350,430,'학습 가중치로 결합','@ Wo [D,D]',ORANGE,h=160)
        text(im,'병합은 재배치 / Output projection은 matmul',1138,602,31)
    elif kind=='residual':
        flow(im,['X','Attention','+','X₁'],t,y=416)
        line(im,(605,480),(605,572),TEAL)
        line(im,(605,572),(1321,572),TEAL)
        arrow(im,(1321,572),(1321,490))
        text(im,'원래 입력을 전달하는 우회 경로',1060,612,30,TEAL)
    elif kind in ('ffn','gate'):
        card(im,478,369,225,'입력','LN(X₁)',TEAL,h=145)
        for yy,body in ((315,'@ Wgate'),(505,'@ Wup')):
            arrow(im,(720,440),(850,yy+30))
            box(im,868,yy-15,275,88,outline=ORANGE)
            text(im,body,1005,yy+29,33,ORANGE)
        box(im,1190,300,250,88,outline=PURPLE)
        text(im,'SiLU',1315,344,34,PURPLE)
        arrow(im,(1148,344),(1184,344))
        arrow(im,(1446,344),(1575,433),PURPLE)
        arrow(im,(1150,534),(1575,469),TEAL)
        box(im,1590,392,156,119,outline=PURPLE)
        text(im,'⊙',1668,450,65,PURPLE)
        text(im,'각 토큰을 독립적으로 변환',1138,613,30,TEAL)
    elif kind=='ledger':
        labels=['Q','K','V','QKᵀ','PV','O','Gate','Up','Down']
        for j,name in enumerate(labels):
            x=482+(j%5)*262; y=299+(j//5)*153
            col=PURPLE if name in ('QKᵀ','PV') else ORANGE
            box(im,x,y,231,112,outline=col)
            text(im,f'{j+1:02}',x+38,y+30,22,MUTED)
            text(im,name,x+116,y+72,34,col)
        text(im,'주황 7개: 학습 가중치 / 보라 2개: activation끼리',1138,624,30)
    elif kind=='totals':
        for j,(label_,body,col) in enumerate([('Attention','4D²',TEAL),('SwiGLU FFN','3DF',PURPLE),('LayerNorm × 2','4D',ORANGE)]):
            card(im,475+j*448,322,415,label_,body,col,h=175)
        text(im,'Params에는 T가 없다 / FLOPs에는 T와 S가 있다',1138,590,33)
    elif kind=='numbers':
        data=[('Attention',FACTS['attention_weights'],TEAL),('SwiGLU FFN',FACTS['ffn_weights'],PURPLE),('LayerNorm × 2',FACTS['norm_params_layer'],ORANGE)]
        for j,(name,value,col) in enumerate(data):
            yy=315+j*104
            text(im,name,478,yy,29,MUTED,left=True,width=275)
            extent=max(3,665*value/FACTS['ffn_weights'])*ease(t/2)
            box(im,785,yy-22,extent,44,col,col,5)
            text(im,f'{value:,}',1685,yy,31,col,width=290)
        text(im,f'한 층 합계  {FACTS["layer_params"]:,}개',1138,617,39,INK)
    elif kind=='modeltotal':
        for j,(name,value,col) in enumerate([('8개 층',FACTS['layer_params']*8,TEAL),('Embedding',FACTS['embedding_params'],ORANGE),('LM head',FACTS['head_params'],PURPLE)]):
            card(im,475+j*449,328,414,name,f'{value:,}',col,h=170)
        text(im,f'+ Final LayerNorm {2*FACTS["D"]:,}개 = {FACTS["total_params"]:,}개',1138,588,35)
    elif kind=='count':
        text(im,'9 × 8 + 1',1138,367,100,TEAL)
        text(im,'73개의 논리적 matmul',1138,505,55,PURPLE)
        text(im,'층당 9개 × 8층 + LM head 1개',1138,602,31,MUTED)
    elif kind=='compare':
        card(im,505,307,600,'Prefill · B=1, T=S=2,048',display('prefill_matmul'),TEAL,h=190)
        card(im,1170,307,600,'Decode · B=1, T=1, S=2,048',display('decode_matmul'),PURPLE,h=190)
        text(im,'동일한 542,183,424 Params / 동일한 73개 matmul',1138,556,33,ORANGE)
        text(im,'계산량 비율을 실행시간 비율로 해석하지 않기',1138,620,28,MUTED)
    elif kind=='eight':
        card(im,500,311,607,'기본 FFN: 선형층 2개','6 + 2 = 8',TEAL,h=197)
        card(im,1171,311,607,'SwiGLU: 선형층 3개','6 + 3 = 9',PURPLE,h=197)
        text(im,'비교 대상: self-attention 하나를 포함한 블록',1138,593,33,MUTED)
    else:
        listlines(im,s['lines'],t,s,y=302,step=90 if len(s['lines'])<4 else 79)


def frame(index,t):
    s=scenes()[index]
    im=background(index).copy()
    draw(im,s,t)
    prev=scenes()[index-1]['count'] if index else 0
    count=s['count'] if t>=s['reveal'][1] else prev
    text(im,f'{count} / 9',299,865,24,TEAL,width=105)
    line(im,(0,1076),(W,1076),EDGE,6)
    line(im,(0,1076),(W*(s['start']+t)/600,1076),TEAL,6)
    fade=min(ease(t/.25),ease((s['duration']-t)/.25))
    if fade<1:
        im=Image.blend(Image.new('RGB',(W,H),BG),im,fade)
    return im


def signature(index):
    h=hashlib.sha256(Path(__file__).read_bytes())
    for p in ('facts.py','facts.json','lesson.py'):
        h.update((ROOT/p).read_bytes())
    h.update(json.dumps(scenes()[index],ensure_ascii=False).encode())
    for f in (FONT,MATH):h.update(Path(f).read_bytes())
    return h.hexdigest()


def render_one(index):
    s=scenes()[index]; out=ROOT/'clips'; out.mkdir(exist_ok=True)
    target=out/f'{index:02}.mp4'; meta=target.with_suffix('.json'); sig=signature(index)
    if target.exists() and meta.exists() and json.loads(meta.read_text()).get('signature')==sig:
        print(f'Cached scene {index+1}',flush=True); return
    part=target.with_suffix('.part.mp4')
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264',
        '-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(part)],stdin=subprocess.PIPE)
    try:
        for n in range(s['frames']):
            proc.stdin.write(frame(index,n/FPS).tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f'Encoding failed: {index}')
    except BaseException:
        proc.kill(); proc.wait(); raise
    part.replace(target)
    meta.write_text(json.dumps(dict(signature=sig,frames=s['frames'])))
    print(f'Rendered {index+1}/{len(scenes())}: {s["title"]}',flush=True)


def render(only=None):
    if only is not None:
        render_one(only)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(render_one,range(len(scenes()))))


def preview(only=None):
    out=ROOT/'preview'; out.mkdir(exist_ok=True)
    indices=[only] if only is not None else list(range(len(scenes())))
    sheet=Image.new('RGB',(4*480,math.ceil(len(indices)/4)*270),BG)
    for j,i in enumerate(indices):
        s=scenes()[i]
        im=frame(i,s['duration']*.78)
        im.save(out/f'{i:02}.png')
        sheet.paste(im.resize((480,270)),((j%4)*480,(j//4)*270))
    sheet.save(out/'contact-sheet.jpg',quality=93)
    print(out/'contact-sheet.jpg')
