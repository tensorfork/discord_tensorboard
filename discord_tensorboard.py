import os
import argparse
import pathlib
import numpy as np
#import cv2
import io
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

# https://stackoverflow.com/a/1398742/9919772

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, required=True)
parser.add_argument("--channel", type=str, required=True)
parser.add_argument("--start", type=int, default=0, required=False)
parser.add_argument("--end", type=int, required=False)
parser.add_argument("--waitsec", default=240, type=int, required=False)
parser.add_argument("--warnsec", default=3600, type=int, required=False)
parser.add_argument("--logstart", default=None, type=str, required=False)
args = parser.parse_args()
discord_token = os.environ['DISCORD_TOKEN']

import discord
import asyncio
import threading

lock = threading.Lock()

def utc():
    from datetime import datetime
    d = datetime.utcnow()
    import calendar
    return calendar.timegm(d.utctimetuple())

import time
os.environ["TZ"] = "US/Pacific"

def timestamp(utc_seconds):
    return time.strftime("%Y-%m-%d %H:%M:%S PST", time.localtime(utc_seconds))

async def send_message(channel, text):
    print("Posting message to {}: {}".format(channel.name, text))
    await channel.send(content=text)

async def send_picture(channel, img, kind='png', name='test', text=None):
    print("Posting picture to {} with text {}".format(channel.name, text))
    f = io.BytesIO()
    img.save(f, kind)
    f.seek(0)
    picture = discord.File(f)
    picture.filename = name + '.' + kind
    await channel.send(content=text, file=picture)

def get_images(event_acc=None, name='fake_images_image_0'):
  if event_acc is None:
    print("Loading event accumulator for {}".format(args.logdir))
    event_acc = event_accumulator.EventAccumulator(args.logdir, size_guidance={'images': 0})
  print("Reloading event accumulator for {}".format(args.logdir))
  event_acc.Reload()
  print("Finished loading event accumulator for {}".format(args.logdir))
  for tag in event_acc.Tags()['images']:
      events = event_acc.Images(tag)

      tag_name = tag.replace('/', '_')
      if tag_name == name:
          for index, event in enumerate(events):
              s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
              bytes_io = bytearray(s)
              img = Image.open(io.BytesIO(bytes_io))
              yield index, img, event

def bot(channel_name, name='test', kind='png'):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    client = discord.Client()
    token = discord_token

    #print("Loading event accumulator")
    #event_acc = event_accumulator.EventAccumulator(args.logdir, size_guidance={'images': 0})
    #event_acc.Reload() # preload
    event_acc = None

    @client.event
    async def on_ready():
        print('Logged on as {0}!'.format(client.user))
        try:
          channel = [x for x in list(client.get_all_channels()) if channel_name == x.name]
          assert len(channel) == 1
          channel = channel[0]
          print(channel)
          import time

          warnevent = 0.0
          while True:
              results = list(sorted([(index, image, event) for index, image, event in get_images(event_acc)]))

              lastevent = utc()
              start_time = 0.0

              with lock:
                  for index, image, event in results:
                      lastevent = event.wall_time
                      if index == 0:
                          start_time = event.wall_time
                      text = "```#{} step {} elapsed {:.2f}m\n{}\n{}```".format(index, event.step, (event.wall_time - start_time)/60.0, timestamp(event.wall_time), args.logdir)
                      print(text)
                      if index >= args.start and (args.end is None or index <= args.end):
                          args.start = index + 1
                          try:
                              await send_picture(channel, image, 'png', text=text)
                          except:
                              import traceback
                              traceback.print_exc()

              now = utc()
              if args.warnsec is not None and now - lastevent > args.warnsec and now - warnevent > args.warnsec:
                  await send_message(channel, text="I've fallen and I can't get up. Please send help. Last event was {:.2f}m ago.".format((now - lastevent)/60.0))
                  warnevent = now

              if args.waitsec is not None and args.waitsec > 0:
                  print("Sleeping for {} secs".format(args.waitsec))
                  time.sleep(args.waitsec)
              else:
                  print("Done. Bye!")
                  print("--start {}".format(args.start))
                  if args.logstart is not None:
                    with open(args.logstart, "w") as f:
                      f.write(str(args.start))
                  import posix
                  posix._exit(args.start)
                  assert False
                  break
        except:
          import traceback
          traceback.print_exc()
        finally:
          await client.logout()

    #@client.event
    #async def on_message(message):
    #    print('Message from {0.author}: {0.content}'.format(message))

    client.run(token)
bot(args.channel)
