# coding:utf-8


import os
import shutil
import copy
import csv
import logging
import multiprocessing
import random
import re
import time
import sys
import traceback
import subprocess
from multiprocessing import Pool

import pprint as Pprint
from pprint import pprint, pformat


class pprint_utf8_C(Pprint.PrettyPrinter):

    def format(self, object, context, maxlevels, level):
        if isinstance(object, unicode):
            return (object.encode('utf8'), True, False)
        return Pprint.PrettyPrinter.format(self, object, context, maxlevels, level)
pprint_utf8 = pprint_utf8_C().pprint

from timeit import Timer
from datetime import datetime
from datetime import timedelta

# import parent module
# import os, sys
# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         os.pardir)
# )

try:
    import chardet
except Exception as e:
    print e

from cStringIO import StringIO

try:
    import ujson
except:
    ujson = None
try:
    import simplejson
except:
    simplejson = None

if ujson is None:
    if simplejson is None:
        import json
        wjson = json
        rjson = json
    else:
        wjson = simplejson
        rjson = simplejson
        json = simplejson
else:
    if simplejson is None:
        wjson = ujson
        rjson = ujson
        json = ujson
    else:
        wjson = ujson
        rjson = simplejson
        json = ujson

import cPickle as pickle
# import pickle

from bs4 import BeautifulSoup as BS
# enhance bs
from lxml import etree


class BS_plus():

    def __init__(self, page_string=None, etree_element=None):
        if page_string is not None:
            self.page_string = page_string
            page_io = StringIO(page_string)
            parser = etree.HTMLParser()
            self.html_tree = etree.parse(page_io, parser)
        else:
            self.html_tree = etree_element

    def xpath(self, path, mode=""):
        records = [BS_plus(etree_element=e)
                           for e in self.html_tree.xpath(path)]
        if len(records) == 0:
            raise Exception("xpath fail to find")

        if mode == "list":
            return records
        else:
            if len(records) == 1:
                return records[0]
            else:
                return records

    def __getattr__(self, key):
        return getattr(self.html_tree, key)

    def __str__(self):
        return self.html_tree.tag

    def __repr__(self):
        return self.html_tree.tag

    def stringify(self):
        return etree.tostring(self.html_tree, pretty_print=True, method="html")

    @property
    def bs(self):
        self.html_bs = BS(self.page_string)
        return self.html_bs

from HTMLParser import HTMLParser


from jinja2 import Template
import requests as REQ
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn

# webdriver
# from selenium import webdriver
# from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.keys import Keys

PATH_JOIN = os.path.join
PATH_EXISTS = os.path.exists
PATH_SPLIT = os.path.split
CUR_PATH = os.path.split(os.path.realpath(__file__))[0]


def CUR_PATH_JOIN(path):
    return PATH_JOIN(CUR_PATH, path)
PARENT_PATH = PATH_SPLIT(CUR_PATH)[0]


def timeit_func(func):
    print "Timer On"
    t1 = Timer(func + "()", "from __main__ import " + func)
    print t1.timeit(1)


def split_seq(seq, n_part, mode="np"):
    assert len(seq) >= n_part
    gap = len(seq) / n_part
    new_seq = []

    cur_index = 0
    while 1:
        rest = len(seq) - cur_index
        if rest == 0:
            break
        elif rest < gap:
            new_seq.append(seq[cur_index:cur_index + rest])
            break
        else:
            new_seq.append(seq[cur_index:cur_index + gap])
        cur_index += gap

    return new_seq


def split_seq_indices(seq, n_part):
    assert len(seq) >= n_part
    gap = len(seq) / n_part
    indices_seq = []

    cur_index = 0
    next_index = 0
    while 1:
        rest = len(seq) - cur_index

        if rest < gap:
            next_index = -1
        else:
            next_index = cur_index + gap

        indices_seq.append([cur_index, next_index])
        cur_index = next_index + 1
        if next_index == -1:
            break

        return indices_seq


def to_unicode(s):
    ts = type(s)
    if not (ts == type("") or ts == type(u"")):
        raise UnicodeError("to_unicode cant convert non string-like object")

    if not isinstance(s, type(u"")):
        encoding = chardet.detect(s)["encoding"]
        if encoding == "ascii":
            try:
                return s.decode("unicode-escape")
            except:
                pass

        # print s,encoding
        return s.decode(encoding)
    else:
        return s


def to_utf8(s):
    s = to_unicode(s)
    return s.encode("utf-8")


def try_until(do_func, until_func, do_params=[], until_params=[], n_interval=3, n_retry=3):
    for i in xrange(n_retry):
        try:
            do_func(do_params)
        except Exception as e:
            until_func(until_params, e)
            time.sleep(n_interval)


def do_until(do_func, until_func, do_params=[], until_params=[]):
    while 1:
        result = do_func(do_params)
        if until_func(result, until_params):
            break


class Dir():

    def __init__(self, name=None, mode="default", init=False, path=None):
        self.mode = mode
        if path is None:
            # self.path = PATH_JOIN(CUR_PATH, name)
            self.path = PATH_JOIN(PARENT_PATH, name)
        else:
            self.path = path
        if name is None:
            name = PATH_SPLIT(self.path)[-1]
        self.name = name
        if PATH_EXISTS(self.path) and init:
            # print "clean "+self.path
            shutil.rmtree(self.path)
        if not PATH_EXISTS(self.path):
            os.makedirs(self.path)

    def init_dir(self):
        shutil.rmtree(self.path)
        os.makedirs(self.path)

    def _config(self):
        if self.mode == "pickle":
            self.save_mode = "wb"
            self.save_func = lambda F, c: pickle.dump(c, F, -1)
            self.load_mode = "rb"
            self.load_func = lambda F: pickle.load(F)
        elif self.mode == "json":
            self.save_mode = "w"
            self.save_func = lambda F, c: wjson.dump(c, F, ensure_ascii=False)
            self.load_mode = "r"
            self.load_func = lambda F: rjson.load(F)
        else:
            self.save_mode = "w"
            self.save_func = lambda F, c: F.write(c)
            self.load_mode = "r"
            self.load_func = lambda F: F.read()

    def set(self, name, file):
        self._config()
        with open(PATH_JOIN(self.path, name), self.save_mode) as F:
            self.save_func(F, file)

    def get(self, name, decoding=None):
        self._config()
        with open(PATH_JOIN(self.path, name), self.load_mode) as F:
            s = self.load_func(F)

            if decoding is not None:
                s.decode(decoding)

            return s

    def delete(self, name):
        if PATH_EXISTS(PATH_JOIN(self.path, name)):
            os.remove(PATH_JOIN(self.path, name))
        else:
            return False

    def has(self, name):
        return PATH_EXISTS(PATH_JOIN(self.path, name))

    def files(self):
        for name in self.filenames():
            yield self.get(name)

    def filenames(self):
        return os.listdir(self.path)

    def path_join(self, name):
        return PATH_JOIN(self.path, name)

    def set_path(self, path, init=False):
        self.path = path
        if not PATH_EXISTS(self.path):
            os.makedirs(self.path)
        if init:
            self.init_dir()
        self.name = PATH_SPLIT(self.path)[-1]


if False:
    DIR_ROOT = Dir(path=PARENT_PATH)
    DIR_MAIN = Dir(path=PATH_JOIN(PARENT_PATH, "main"))
    DIR_CONF = Dir(path=PATH_JOIN(PARENT_PATH, "conf"))
    DIR_LOG = Dir(path=PATH_JOIN(PARENT_PATH, "log"))
    DIR_TEST = Dir(path=PATH_JOIN(PARENT_PATH, "test"))


def Dir_Root(name, mode="default", init=False):
    return Dir(path=PATH_JOIN(PARENT_PATH, name), mode=mode, init=init)


def Dir_Conf(name, mode="default", init=False):
    try:
        return Dir(path=PATH_JOIN(DIR_CONF.path, name), mode=mode, init=init)
    except NameError:
        pass


def Dir_Log(name, mode="default", init=False):
    try:
        return Dir(path=PATH_JOIN(DIR_LOG.path, name), mode=mode, init=init)
    except NameError:
        pass


def Dir_Test(name, mode="default", init=False):
    try:
        return Dir(path=PATH_JOIN(DIR_TEST.path, name), mode=mode, init=init)
    except NameError:
        pass


# LOGGER

_DIR_LOG_INFO = Dir_Log("log_info", init=True)
# _DIR_LOG_INFO.set_path("/data/log/events_generator_log", init=True)

_LOGGER_DICT = {}


def _setup_logger_local(logger_name):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', '%a, %d %b %Y %H:%M:%S')

    file_handler = logging.FileHandler(
        _DIR_LOG_INFO.path_join(logger_name), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)

    return logger

# import raven
# raven_client =
# raven.Client("http://f98b7cbe8fdc4ce782d50b270e2f00ac:45c748a0f9ca4992aad98ecce4852f57@121.41.111.181/4")


class raven_client_pesuedo():

    def captureException(self):
        pprint(traceback.format_exception(*sys.exc_info()))
raven_client = raven_client_pesuedo()

# DEPRECATED
LOGGER_RAVEN_FLAG = False
__RAVEN_INIT = False


def _setup_logger_raven(logger_name):
    global __RAVEN_INIT
    if not __RAVEN_INIT:
        from raven import Client
        from raven.handlers.logging import SentryHandler
        client = Client("https://")
        raven_handler = SentryHandler(client)
        __RAVEN_INIT = True
        from raven.conf import setup_logging
        setup_logging(raven_handler)

    logger = logging.getLogger(logger_name)
    return logger


# if use raven
if LOGGER_RAVEN_FLAG:
    _setup_logger = _setup_logger_raven
else:
    _setup_logger = _setup_logger_local
# DEPRECATED


def Logger(logger_name="_root"):

    def _print_log(msg="", level="info"):
        if logger_name not in _LOGGER_DICT.keys():
            logger = _setup_logger(logger_name)
            _LOGGER_DICT[logger_name] = logger

        logger = _LOGGER_DICT[logger_name]
        if level == "error":
            logger.error(msg, exc_info=True)
        else:
            getattr(logger, level.lower())(msg)

    return _print_log


# class Logger():

#     _path_join = _DIR_LOG_INFO.path_join

#     def __init__(self, logger_name):
#         self._logger_name=logger_name
#         self._logger = self._setup_logger(logger_name)

#     def _setup_logger(self, logger_name):
#         logger = logging.getLogger(logger_name)
# formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d]
# %(levelname)s %(message)s', '%a, %d %b %Y %H:%M:%S' )

#         file_handler = logging.FileHandler(self._path_join(logger_name),mode="a",encoding="utf-8")
#         file_handler.setFormatter(formatter)

#         stream_handler = logging.StreamHandler(sys.stdout)
#         stream_handler.setFormatter(formatter)

#         logger.addHandler(file_handler)
#         logger.addHandler(stream_handler)

#         logger.setLevel(logging.INFO)

#         return logger

#     def print_log(self, msg="", level="info"):
#         logger=self._logger
#         getattr(logger, level.lower())(msg)

#     def __call__(self, *args, **kwargs):
#         self.print_log(*args, **kwargs)

def to_pep8(dir_path=CUR_PATH):
    import autopep8

    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py") and "Utils_" not in filename:
                filepath = PATH_JOIN(root, filename)

                try:
                    with open(filepath, "r") as F:
                        content = autopep8.fix_code(to_unicode(F.read()))

                    with open(filepath, "w") as F:
                        F.write(to_utf8(to_unicode(content)))

                except Exception as e:
                    print e


def __to_pep8(dir_path=CUR_PATH):
    import autopep8
    import glob

    def clean_dir_main():
        d = DIR_MAIN
        for fn in d.filenames():
            if fn.endswith(".bak"):
                d.delete(fn)

    def _backup_dir(dd):
        dir_backup = Dir_Root("__backup_{dirname}".format(dirname=dd.name))
        shutil.rmtree(dir_backup.path)
        shutil.copytree(dd.path, dir_backup.path)
        print "backup_{dirname} ok".format(dirname=dd.name)

    def _to_pep8(dd):
        # dir_backup = Dir_Root("__backup")

        for fn in dd.filenames():
            if fn.endswith(".py"):
                try:
                    content = autopep8.fix_code(to_unicode(dd.get(fn)))
                    dd.set(fn, to_utf8(to_unicode(content)))
                except Exception as e:
                    print e
                    dd.set(fn, DIR_MAIN.get(fn))

    _backup_dir(DIR_MAIN)
    _to_pep8(DIR_MAIN)

    _backup_dir(DIR_ROOT)
    _to_pep8(DIR_ROOT)


def sleep_schedule(year=None, month=None, day=None, hour=None, minute=None, second=None, date=None):
    """
    wakeup_time is like this : "2015,9,18,19,35,30 - year,month,day,hour,minute,second"
    """

    ll = ["year", "month", "day", "hour", "minute", "second"]
    if date is None:
        schedule_dict = {"year": year, "month": month, "day": day,
            "hour": hour, "minute": minute, "second": second}
    else:
        schedule_dict = {"year": date.year, "month": date.month, "day": date.day,
            "hour": date.hour, "minute": date.minute, "second": date.second}

    cur_time = datetime.now()
    cyear = cur_time.year
    cmonth = cur_time.month
    cday = cur_time.day
    chour = cur_time.hour
    cminute = cur_time.minute
    csecond = cur_time.second

    temp = dict(year=cyear, month=cmonth, day=cday,
                hour=chour, minute=cminute, second=csecond)
    for key in temp.keys():
        if schedule_dict[key] is None:
            schedule_dict[key] = temp[key]

    next_time = datetime(*[int(schedule_dict[key]) for key in ll])
    sleep_seconds = (next_time - cur_time).total_seconds()

    time.sleep(sleep_seconds)


def sleep_aclock(hour=12, minute=0, day_interval=1):
    today_aclock = datetime.now()
    today_aclock = datetime(
        today_aclock.year, today_aclock.month, today_aclock.day, hour, minute, 0)

    if datetime.now() < today_aclock:
        day_interval -= 1

    next_date = datetime.now() + timedelta(days=day_interval)
    next_date = datetime(next_date.year, next_date.month,
                         next_date.day, hour, minute, 0)

    # print next_date
    sleep_schedule(date=next_date)


def text_clean(text):
    text = re.sub(r'[\x00-\x7F]+', u' ', text)
    text = text.replace(u" ", u" ")
    text = text.strip()
    return text


def is_string_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_blank_line(s):
    ns = s.replace(" ", "").replace("	","")
    for c in ns:
        if c != "":
            return False
    return True


def __main():
    # sleep_aclock(hour=10,minute=57)
    # print "wakeup you asshole"

    to_pep8()


if __name__ == '__main__':
    __main()
