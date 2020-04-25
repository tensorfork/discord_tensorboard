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
parser.add_argument("--channel", type=str, required=False)
parser.add_argument("--start", type=int, default=0, required=False)
parser.add_argument("--end", type=int, required=False)
parser.add_argument("--waitsec", default=0, type=int, required=False)
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

biggan_defaults = dict([
 ['AdamOptimizer.beta1', 0.0],
 ['AdamOptimizer.beta2', 0.999],
 ['AdamOptimizer.epsilon', 1e-08],
 ['AdamOptimizer.use_locking', False],
 ['BigGanResNetBlock.add_shortcut', True],
 ['conditional_batch_norm.use_bias', False],
 ['cross_replica_moments.group_size', None],
 ['cross_replica_moments.parallel', True],
 ['D.batch_norm_fn', None],
 ['D.layer_norm', False],
 ['D.spectral_norm', True],
 ['dataset.name', ""],
 ['dataset.seed', 547],
 ['resnet_biggan.Discriminator.blocks_with_attention', ""],
 ['resnet_biggan.Discriminator.ch', ''],
 ['resnet_biggan.Discriminator.channel_multipliers', None],
 ['resnet_biggan.Discriminator.project_y', True],
 ['G.batch_norm_fn', '@conditional_batch_norm'],
 ['G.spectral_norm', True],
 ['resnet_biggan.Generator.blocks_with_attention', ""],
 ['resnet_biggan.Generator.ch', ''],
 ['resnet_biggan.Generator.channel_multipliers', None],
 ['resnet_biggan.Generator.embed_bias', False],
 ['resnet_biggan.Generator.embed_y', True],
 ['resnet_biggan.Generator.embed_y_dim', 128],
 ['resnet_biggan.Generator.embed_z', False],
 ['resnet_biggan.Generator.hierarchical_z', True],
 ['loss.fn', '@hinge'],
 ['ModularGAN.conditional', True],
 ['ModularGAN.d_lr', ''],
 ['ModularGAN.d_optimizer_fn', '@tf.train.AdamOptimizer'],
 ['ModularGAN.deprecated_split_disc_calls', False],
 ['ModularGAN.ema_decay', 0.9999],
 ['ModularGAN.ema_start_step', 40000],
 ['ModularGAN.experimental_force_graph_unroll', False],
 ['ModularGAN.experimental_joint_gen_for_disc', False],
 ['ModularGAN.fit_label_distribution', False],
 ['ModularGAN.g_lr', ''],
 ['ModularGAN.g_optimizer_fn', '@tf.train.AdamOptimizer'],
 ['ModularGAN.g_use_ema', True],
 ['normal.mean', 0.0],
 ['normal.seed', None],
 ['options.architecture', 'resnet_biggan_arch'],
 ['options.batch_size', ''],
 ['options.d_flood', ''],
 ['options.datasets', ""],
 ['options.description', 'Describe your GIN config. (This appears in the tensorboard text tab.)'],
 ['options.disc_iters', 2],
 ['options.discriminator_normalization', None],
 ['options.g_flood', ''],
 ['options.gan_class', '@ModularGAN'],
 ['options.image_grid_height', '*exclude'],
 ['options.image_grid_resolution', '*exclude'],
 ['options.image_grid_width', '*exclude'],
 ['options.labels', None],
 ['options.lamba', 1],
 ['options.model_dir', "*exclude"],
 ['options.num_classes', 1000],
 ['options.random_labels', ''],
 ['options.training_steps', 250000],
 ['options.transpose_input', False],
 ['options.z_dim', ''],
 ['penalty.fn', '@no_penalty'],
 ['replace_labels.file_pattern', None],
 ['run_config.iterations_per_loop', 250],
 ['run_config.keep_checkpoint_every_n_hours', 1],
 ['run_config.keep_checkpoint_max', 10],
 ['run_config.save_checkpoints_steps', 250],
 ['run_config.single_core', False],
 ['run_config.tf_random_seed', None],
 ['spectral_norm.epsilon', 1e-12],
 ['spectral_norm.singular_value', 'auto'],
 ['standardize_batch.decay', 0.9],
 ['standardize_batch.epsilon', 1e-05],
 ['standardize_batch.use_cross_replica_mean', None],
 ['standardize_batch.use_moving_averages', False],
 ['train_imagenet_transform.crop_method', 'random'],
 ['weights.initializer', 'orthogonal'],
 ['z.distribution_fn', '@tf.random.normal'],
 ['z.maxval', 1.0],
 ['z.minval', -1.0],
 ['z.stddev', 1.0]
 ])

def _get_tensors(event_acc, names):
  if not isinstance(names, list):
    names = [names]
  tags = event_acc.Tags()
  for tensor in tags.get('tensors', []):
    if tensor in names:
      for event in event_acc.tensors.Items(tensor):
        props = dict([(k.name, v) for k, v in event.tensor_proto.ListFields()])
        props['event'] = event
        props['tensor'] = tensor
        yield props

def get_tensors(event_acc, names=['gin/operative_config'], step=None):
  results = list(sorted(_get_tensors(event_acc, names), key=lambda x: x['event'].step))
  if step is not None and len(results) > 0:
    best = None
    for result in results:
      if best is None:
        best = result
      if result['event'].step <= step:
        if result['event'].step >= best['event'].step:
          best = result
    return best
  return results

def get_string_val(x, unset=None):
  if x is None:
    return unset
  return x['string_val'][0].decode('utf8')

def get_config(event_acc, step, description=None, match=None, exclude=None):
  result = get_tensors(event_acc, 'gin/operative_config', step=step)
  if result is None:
    return None
  cfg = get_string_val(result)
  if cfg is None:
    cfg = "# No config"
  cfg = cfg.replace('\r', '').replace('\\\n        ', '')
  cfg = "_config.step = {}\n{}".format(result['event'].step, cfg)
  if match is not None:
    if exclude is None:
      exclude = []
    if not isinstance(match, list):
      match = [match]
    if not isinstance(exclude, list):
      exclude = [exclude]
    cfg = '\n'.join([x for x in cfg.splitlines() if any([x.lstrip().startswith(y) for y in match]) and not any([x.lstrip().startswith(y) for y in exclude])])
    return cfg
  if description is not None:
    description = " " + description
  else:
    description = ""
  cfg = "{} at step {}{}\n{}".format(result['tensor'], result['event'].step, description, cfg)
  return cfg

def get_settings(event_acc, step):
  import pdb; pdb.set_trace()
  return [x.strip().split(' = ', 1) for x in get_config(event_acc, step=step, match='', exclude=''.split()).splitlines() if not x.strip().startswith('#') and len(x.strip()) > 0]

import gin

def _get_settings(event_acc, step):
  cfg = get_config(event_acc, step=step, match='', exclude=''.split())
  parser = gin.config_parser.ConfigParser(cfg, gin.config.ParserDelegate(skip_unknown=True))
  for statement in parser:
    k = '{}.{}'.format(statement.selector, statement.arg_name)
    v = statement.value
    if hasattr(v, 'selector'):
      v = '@{}'.format(v.selector)
    yield k, v

def get_settings(event_acc, step):
  return [x for x in _get_settings(event_acc, step)]

def get_settings_diff(event_acc, step, exclude=biggan_defaults):
  return [(k, v) for k, v in get_settings(event_acc, step) if exclude.get(k) != v and exclude.get(k) != '*exclude' or k.startswith('*')]

import json

def get_description(event_acc, step):
  #result = get_config(event_acc, step=step, match='options', exclude='options.image_ options.transpose_input options.training_steps'.split())
  result = get_settings_diff(event_acc, step)
  if result is None:
    return ""
  return json.dumps(dict(result))

def get_images(event_acc, name='fake_images_image_0'):
  tags = event_acc.Tags()
  for tag in tags['images']:
      events = event_acc.Images(tag)
      tag_name = tag.replace('/', '_')
      if tag_name == name:
          for index, event in enumerate(events):
              s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
              bytes_io = bytearray(s)
              img = Image.open(io.BytesIO(bytes_io))
              yield index, img, event

def truncate_text(text, size=3000):
  if text is None:
    return text
  if len(text) < size:
    return text
  code = False
  n = size - 1
  m = 0
  if text.startswith('```') and text.endswith('```'):
    text = text[3:-3]
    m = 6
    code = True
  text = text[0:max(0, n-m-3)] + '...'
  if code:
    text = '```' + text + '```'
  print(repr(text))
  assert len(text) < size or size <= 3+m
  return text

async def send_message(channel, text):
  text = truncate_text(text)
  if channel is None:
    print("Posting message: {}".format(text))
  else:
    print("Posting message to {}: {}".format(channel.name, text))
    await channel.send(content=text)

async def send_picture(channel, img, kind='jpg', name='test', text=None):
  text = truncate_text(text)
  if channel is None:
    print("Posting picture with text {}".format(text))
  else:
    print("Posting picture to {} with text {}".format(channel.name, text))
    f = io.BytesIO()
    if kind.lower() in ['jpg', 'jpeg']:
      if '=' in kind:
        kind, quality = kind.split('=')
        quality = int(quality)
      else:
        quality = 95
      img.save(f, 'JPEG', quality=quality)
    else:
      img.save(f, kind)
    f.seek(0)
    picture = discord.File(f)
    picture.filename = name + '.' + kind
    await channel.send(content=text, file=picture)

def bot(name='test', kind='jpg'):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    client = discord.Client()
    token = discord_token

    print("Loading event accumulator")
    event_acc = event_accumulator.EventAccumulator(args.logdir, size_guidance={'images': 0})
    event_acc.Reload() # preload

    @client.event
    async def on_ready():
        print('Logged on as {0}!'.format(client.user))
        try:
          channel = None if args.channel is None else [x for x in list(client.get_all_channels()) if args.channel == x.name]
          if channel is not None:
            assert len(channel) == 1
            channel = channel[0]
            print(channel)

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
                      desc = get_description(event_acc, step=event.step)
                      if len(desc.strip()) > 0:
                        desc = '\n' + desc.strip()
                      text = "```#{} step {} elapsed {:.2f}m\n{}\n{}{}```".format(index, event.step, (event.wall_time - start_time)/60.0, timestamp(event.wall_time), args.logdir, desc)
                      print(text)
                      if index >= args.start and (args.end is None or index <= args.end):
                          args.start = index + 1
                          try:
                              await send_picture(channel, image, 'jpg', text=text)
                          except:
                              import traceback
                              traceback.print_exc()

              now = utc()
              if args.warnsec is not None and now - lastevent > args.warnsec and now - warnevent > args.warnsec:
                  await send_message(channel, text="I've fallen and I can't get up. Please send help for logdir {}. Last update was {:.2f}m ago.".format(args.logdir, (now - lastevent)/60.0))
                  warnevent = now

              if args.waitsec is not None and args.waitsec > 0:
                  print("Sleeping for {} secs".format(args.waitsec))
                  await asyncio.sleep(args.waitsec)
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
              print('Reloading events for {}'.format(args.logdir))
              event_acc.Reload()
              print('Reloaded events for {}'.format(args.logdir))
        except:
          import traceback
          traceback.print_exc()
        finally:
          await client.logout()

    #@client.event
    #async def on_message(message):
    #    print('Message from {0.author}: {0.content}'.format(message))

    client.run(token)
bot()
