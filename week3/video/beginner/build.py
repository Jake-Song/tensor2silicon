"""Build a new beginner lesson; no existing video clips are inputs.

Commands: audio, sync, preview, render, assemble, check, bundle.
Use an edge-tts environment for audio and a Pillow environment for rendering.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import wave
import zipfile

ROOT = Path(__file__).resolve().parent
FINAL = ROOT.parent / 'week3-beginner.ko.mp4'
PREVIEW = ROOT.parent / 'week3-beginner-preview.ko.mp4'
VOICE = 'ko-KR-SunHiNeural'


def run(args):
    subprocess.run([str(x) for x in args], check=True)


def timeline():
    return json.loads((ROOT / 'timeline.json').read_text())


def write_script(scenes):
    parts = ['# 주방 비유로 배우는 GPU 계산과 데이터 이동',
             '초보자용 새 대본 · 26개 장면 · 여성 목소리 SunHi · 10분']
    for s in scenes:
        seconds = int(s['start'])
        stamp = f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'
        parts.append(f'## {stamp} · {s["title"]}\n\n{s["narration"]}\n\n화면 핵심: {s["takeaway"]}')
    (ROOT/'script.ko.md').write_text('\n\n'.join(parts)+'\n')


def audio():
    # Reuse only speech generation and timing utilities, never footage or old audio.
    spec = importlib.util.spec_from_file_location('speech_pipeline', ROOT.parent / 'intuition' / 'produce.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT.parent / 'intuition'))
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SCENES = [dict(s, equation=s['takeaway'], note='', labels=[])
                     for s in json.loads((ROOT / 'lesson.json').read_text())]
    asyncio.run(module.audio())
    sync()


def sync():
    scenes = timeline()
    for s in scenes:
        i = s['index']
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', ROOT/'audio'/f'{i:02}.mp3',
             '-af', f'aresample=48000,atempo={s["tempo"]},asetpts=N/SR/TB,adelay=350:all=1,apad,atrim=end_sample={s["frames"]*1600},asetpts=N/SR/TB',
             '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', ROOT/'audio'/f'{i:02}-timed.wav'])
    listing = ROOT/'audio'/'concat.txt'
    listing.write_text('\n'.join(f"file '{i:02}-timed.wav'" for i in range(len(scenes))))
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', listing, '-c', 'copy', ROOT/'audio'/'narration.wav'])
    groups = []
    for s in scenes:
        if not groups or groups[-1]['title'] != s['chapter']:
            groups.append(dict(title=s['chapter'], start=round(s['start']*1000)))
    lines = [';FFMETADATA1', 'title=주방 비유로 배우는 GPU 계산과 데이터 이동', 'language=kor']
    for i, g in enumerate(groups):
        end = groups[i+1]['start'] if i+1 < len(groups) else 600000
        lines += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={g["start"]}', f'END={end}', f'title={g["title"]}']
    (ROOT/'chapters.ffmetadata').write_text('\n'.join(lines)+'\n')
    write_script(scenes)
    print('Synced new narration and chapter timing', flush=True)


def assemble():
    import visuals
    scenes = timeline()
    for s in scenes:
        i = s['index']
        assert json.loads((ROOT/'clips'/f'{i:02}.json').read_text())['signature'] == visuals.signature(i)
    listing = ROOT/'clips'/'concat.txt'
    listing.write_text('\n'.join(f"file '{i:02}.mp4'" for i in range(len(scenes))))
    style = 'FontName=UnDotum,FontSize=15,PrimaryColour=&H00FFFFFF,OutlineColour=&H00251F17,BorderStyle=1,Outline=2,Shadow=0,MarginV=22,Alignment=2'
    part = FINAL.with_suffix('.part.mp4')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing,
         '-i', ROOT/'audio'/'narration.wav', '-i', ROOT/'chapters.ffmetadata',
         '-map', '0:v', '-map', '1:a', '-map_metadata', '2', '-map_chapters', '2',
         '-vf', f"subtitles='{ROOT/'week3.ko.srt'}':force_style='{style}'",
         '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-c:v', 'libx264', '-threads', '4',
         '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
         '-ar', '48000', '-metadata:s:a:0', 'language=kor', '-t', '600', '-movflags', '+faststart', part])
    part.replace(FINAL)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', FINAL, '-t', '60',
         '-map_chapters', '-1', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '19',
         '-c:a', 'aac', '-movflags', '+faststart', PREVIEW])
    print(FINAL, flush=True)


def check():
    import visuals
    scenes = timeline()
    lesson = json.loads((ROOT/'lesson.json').read_text())
    cursor = 0
    for s, original in zip(scenes, lesson, strict=True):
        i = s['index']
        assert s['narration'] == original['narration']
        assert abs(s['start'] - cursor/30) < 1e-8
        assert s['audio_duration']/s['tempo']+.35 < s['duration']
        meta = json.loads((ROOT/'audio'/f'{i:02}.json').read_text())
        assert meta['voice'] == VOICE and meta['boundaries']
        assert meta['digest'] == hashlib.sha256((VOICE+s['narration']).encode()).hexdigest()
        clip = json.loads((ROOT/'clips'/f'{i:02}.json').read_text())
        assert clip['signature'] == visuals.signature(i) and clip['frames'] == s['frames']
        cursor += s['frames']
    assert cursor == 18000
    with wave.open(str(ROOT/'audio'/'narration.wav')) as f:
        assert f.getnframes() == 600*48000 and f.getframerate() == 48000
    def seconds(value):
        h, m, s = value.replace(',', '.').split(':')
        return 3600*int(h)+60*int(m)+float(s)
    previous = 0
    for block in (ROOT/'week3.ko.srt').read_text().strip().split('\n\n'):
        lines = block.splitlines()
        a, b = map(seconds, lines[1].split(' --> '))
        assert previous <= a < b <= 600 and lines[2].strip()
        previous = b
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_format', '-show_chapters', '-of', 'json', str(FINAL)]))
    v = next(s for s in data['streams'] if s['codec_type'] == 'video')
    assert (v['width'], v['height'], v['r_frame_rate'], int(v['nb_frames'])) == (1920,1080,'30/1',18000)
    assert any(s['codec_type'] == 'audio' for s in data['streams'])
    assert abs(float(data['format']['duration'])-600) < .1
    assert len(data['chapters']) == len({s['chapter'] for s in scenes})
    for chapter, group in zip(data['chapters'], [s for i,s in enumerate(scenes)
        if i==0 or s['chapter']!=scenes[i-1]['chapter']], strict=True):
        assert abs(float(chapter['start_time'])-group['start']) < .002
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', FINAL, '-f', 'null', '-'])
    report = dict(voice=VOICE, scenes=len(scenes), duration=600, resolution=[1920,1080], fps=30,
                  frames=18000, chapters=len(data['chapters']), full_decode='passed',
                  source='new beginner script and newly rendered scenes; no old video inputs',
                  tempo=scenes[0]['tempo'], sha256=hashlib.sha256(FINAL.read_bytes()).hexdigest())
    (ROOT/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('PASS: new script, female voice, narration timing, subtitles, current renders, 1080p30, 600s, chapters, full decode', flush=True)


def bundle():
    files = ['build.py', 'visuals.py', 'lesson.json', 'script.ko.md', 'timeline.json',
             'week3.ko.srt', 'chapters.ffmetadata', 'validation.json', 'README.md', 'audio/narration.wav']
    files += [f'audio/{i:02}.{ext}' for i in range(len(timeline())) for ext in ('mp3','json')]
    with zipfile.ZipFile(ROOT.parent/'week3-beginner-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(ROOT/name, 'week3-video/beginner/'+name)
        for name in ['produce.py','lesson.py']:
            z.write(ROOT.parent/'intuition'/name, 'week3-video/intuition/'+name)
    print('Source bundle saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audio','sync','preview','render','assemble','check','bundle'])
    parser.add_argument('--scene', type=int)
    args = parser.parse_args()
    if args.command in ('preview','render'):
        import visuals
        if args.command == 'render':
            visuals.render(args.scene)
        else:
            visuals.preview()
    else:
        globals()[args.command]()
