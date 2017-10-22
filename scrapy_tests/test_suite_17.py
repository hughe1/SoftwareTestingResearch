import unittest
from scrapy.http import Request
from scrapy.item import BaseItem
from scrapy.utils.spider import iterate_spider_output, iter_spider_classes

from scrapy.spiders import CrawlSpider

import unittest

from scrapy.downloadermiddlewares.ajaxcrawl import AjaxCrawlMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request, HtmlResponse, Response
from scrapy.utils.test import get_crawler

__doctests__ = ['scrapy.downloadermiddlewares.ajaxcrawl']
import unittest

from scrapy.http import Request
from scrapy.downloadermiddlewares.httpauth import HttpAuthMiddleware
from scrapy.spiders import Spider

import unittest
import copy

from scrapy.http import Headers
import hashlib
import tempfile
import unittest
import shutil

from scrapy.dupefilters import RFPDupeFilter
from scrapy.http import Request
from scrapy.utils.python import to_bytes

from unittest import TestCase

from scrapy.downloadermiddlewares.defaultheaders import DefaultHeadersMiddleware
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
from scrapy.utils.python import to_bytes

from twisted.trial import unittest
from twisted.internet import reactor, defer
from twisted.python.failure import Failure

from scrapy.utils.defer import mustbe_deferred, process_chain, \
    process_chain_both, process_parallel, iter_errback

from six.moves import xrange

import logging
from unittest import TestCase

from testfixtures import LogCapture
from twisted.trial.unittest import TestCase as TrialTestCase
from twisted.internet import defer

from scrapy.utils.test import get_crawler
from tests.mockserver import MockServer
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.spidermiddlewares.httperror import HttpErrorMiddleware, HttpError
from scrapy.settings import Settings

import os
import unittest
from six.moves.urllib.parse import urlparse

from scrapy.http import Response, TextResponse, HtmlResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.response import (response_httprepr, open_in_browser,
                                   get_meta_refresh, get_base_url, response_status_message)

__doctests__ = ['scrapy.utils.response']

from twisted.internet import defer
from twisted.trial.unittest import TestCase
from scrapy.utils.test import get_crawler
from tests.spiders import FollowAllSpider, ItemSpider, ErrorSpider
from tests.mockserver import MockServer

import unittest
import warnings
from six.moves import reload_module

import json
import unittest
import datetime
from decimal import Decimal

from twisted.internet import defer

from scrapy.utils.serialize import ScrapyJSONEncoder
from scrapy.http import Request, Response

import gc
import functools
import operator
import unittest
from itertools import count
import platform
import six

from scrapy.utils.python import (
    memoizemethod_noargs, binary_is_text, equal_attributes,
    WeakKeyCache, stringify_dict, get_func_args, to_bytes, to_unicode,
    without_none_values)

__doctests__ = ['scrapy.utils.python']

import unittest

from scrapy.settings import BaseSettings
from scrapy.utils.conf import build_component_list, arglist_to_dict

# -*- coding: utf-8 -*-
import unittest

import six
from six.moves.urllib.parse import urlparse

from scrapy.spiders import Spider
from scrapy.utils.url import (url_is_from_any_domain, url_is_from_spider,
                              add_http_if_no_scheme, guess_scheme,
                              parse_url, strip_url)

__doctests__ = ['scrapy.utils.url']

from unittest import TestCase

from scrapy.spidermiddlewares.urllength import UrlLengthMiddleware
from scrapy.http import Response, Request
from scrapy.spiders import Spider

import os
from datetime import datetime
import shutil
from twisted.trial import unittest

from scrapy.extensions.spiderstate import SpiderState
from scrapy.spiders import Spider
from scrapy.exceptions import NotConfigured
from scrapy.utils.test import get_crawler

from importlib import import_module
from twisted.trial import unittest
import json
import logging

from testfixtures import LogCapture
from twisted.internet import defer
from twisted.trial.unittest import TestCase

from scrapy.http import Request
from scrapy.crawler import CrawlerRunner
from scrapy.utils.python import to_unicode
from tests.spiders import FollowAllSpider, DelaySpider, SimpleSpider, \
    BrokenStartRequestsSpider, SingleRequestSpider, DuplicateStartRequestsSpider
from tests.mockserver import MockServer

import unittest

from scrapy.downloadermiddlewares.downloadtimeout import DownloadTimeoutMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.utils.test import get_crawler

import gzip
import inspect
import warnings
from io import BytesIO

from testfixtures import LogCapture
from twisted.trial import unittest

from scrapy import signals
from scrapy.settings import Settings
from scrapy.http import Request, Response, TextResponse, XmlResponse, HtmlResponse
from scrapy.spiders.init import InitSpider
from scrapy.spiders import Spider, BaseSpider, CrawlSpider, Rule, XMLFeedSpider, \
    CSVFeedSpider, SitemapSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.trackref import object_ref
from scrapy.utils.test import get_crawler

from tests import mock

from __future__ import print_function
import unittest
from scrapy.http import Request
from scrapy.utils.request import request_fingerprint, _fingerprint_cache, \
    request_authenticate, request_httprepr
import unittest
import os
import tempfile
import shutil
import contextlib
from scrapy.utils.project import data_path

import os
import sys
import subprocess
import tempfile
from time import sleep
from os.path import exists, join, abspath
from shutil import rmtree, copytree
from tempfile import mkdtemp
from contextlib import contextmanager

from twisted.trial import unittest
from twisted.internet import defer

import scrapy
from scrapy.utils.python import to_native_str
from scrapy.utils.python import retry_on_eintr
from scrapy.utils.test import get_testenv
from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest

from unittest import TestCase

from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import unittest
from scrapy.linkextractors.regex import RegexLinkExtractor
from scrapy.http import HtmlResponse
from scrapy.link import Link
from scrapy.linkextractors.htmlparser import HtmlParserLinkExtractor
from scrapy.linkextractors.sgml import SgmlLinkExtractor, BaseSgmlLinkExtractor
from tests import get_testdata

from tests.test_linkextractors import Base

from unittest import TextTestResult

from twisted.trial import unittest

from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.item import Item, Field
from scrapy.contracts import ContractsManager
from scrapy.contracts.default import (
    UrlContract,
    ReturnsContract,
    ScrapesContract,
)

import sys
from twisted.trial import unittest
from twisted.internet import defer

import scrapy
from scrapy.utils.testproc import ProcessTest

from unittest import TestCase, main
from scrapy.http import Response, XmlResponse
from scrapy.downloadermiddlewares.decompression import DecompressionMiddleware
from scrapy.spiders import Spider
from tests import get_testdata
from scrapy.utils.test import assert_samelines

import unittest

from scrapy.utils.console import get_shell_embed_func
try:
    import bpython
    bpy = True
    del bpython
except ImportError:
    bpy = False
try:
    import IPython
    ipy = True
    del IPython
except ImportError:
    ipy = False
import os
import random
import time
import hashlib
import warnings
from tempfile import mkdtemp
from shutil import rmtree
from six.moves.urllib.parse import urlparse
from six import BytesIO

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.pipelines.files import FilesPipeline, FSFilesStore, S3FilesStore, GCSFilesStore
from scrapy.item import Item, Field
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.utils.python import to_bytes
from scrapy.utils.test import assert_aws_environ, get_s3_content_and_delete
from scrapy.utils.test import assert_gcs_environ, get_gcs_content_and_delete
from scrapy.utils.boto import is_botocore

from tests import mock

from twisted.trial import unittest

from scrapy.settings import Settings
from scrapy.exceptions import NotConfigured
from scrapy.middleware import MiddlewareManager
import six
from __future__ import absolute_import
import re
import json
import marshal
import tempfile
import unittest
from io import BytesIO
from datetime import datetime
from six.moves import cPickle as pickle

import lxml.etree
import six

from scrapy.item import Item, Field
from scrapy.utils.python import to_unicode
from scrapy.exporters import (
    BaseItemExporter, PprintItemExporter, PickleItemExporter, CsvItemExporter,
    XmlItemExporter, JsonLinesItemExporter, JsonItemExporter,
    PythonItemExporter, MarshalItemExporter
)

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re
from twisted.internet import reactor, error
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred
from twisted.python import failure
from twisted.trial import unittest
from scrapy.downloadermiddlewares.robotstxt import (RobotsTxtMiddleware,
                                                    logger as mw_module_logger)
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request, Response, TextResponse
from scrapy.settings import Settings
from tests import mock

import unittest

from scrapy.utils.http import decode_chunked_transfer
import os
import hashlib
import random
import warnings
from tempfile import mkdtemp, TemporaryFile
from shutil import rmtree

from twisted.trial import unittest

from scrapy.item import Item, Field
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.python import to_bytes

skip = False
try:
    from PIL import Image
except ImportError as e:
    skip = 'Missing Python Imaging Library, install https://pypi.python.org/pypi/Pillow'
else:
    encoders = set(('jpeg_encoder', 'jpeg_decoder'))
    if not encoders.issubset(set(Image.core.__dict__)):
        skip = 'Missing JPEG encoders'

from six.moves.urllib.parse import urlparse
import unittest

from io import BytesIO
from unittest import TestCase, SkipTest
from os.path import join
from gzip import GzipFile

from scrapy.spiders import Spider
from scrapy.http import Response, Request, HtmlResponse
from scrapy.downloadermiddlewares.httpcompression import HttpCompressionMiddleware, \
    ACCEPTED_ENCODINGS
from scrapy.responsetypes import responsetypes
from scrapy.utils.gz import gunzip
from tests import tests_datadir
from w3lib.encoding import resolve_encoding


SAMPLEDIR = join(tests_datadir, 'compressed')

FORMAT = {
        'gzip': ('html-gzip.bin', 'gzip'),
        'x-gzip': ('html-gzip.bin', 'gzip'),
        'rawdeflate': ('html-rawdeflate.bin', 'deflate'),
        'zlibdeflate': ('html-zlibdeflate.bin', 'deflate'),
        'br': ('html-br.bin', 'br')
        }

import unittest
import warnings
import six

from scrapy.link import Link

import unittest
from os.path import join

from w3lib.encoding import html_to_unicode

from scrapy.utils.gz import gunzip, is_gzipped
from scrapy.http import Response, Headers
from tests import tests_datadir

SAMPLEDIR = join(tests_datadir, 'compressed')

import pickle

from queuelib.tests import test_queue as t
from scrapy.squeues import MarshalFifoDiskQueue, MarshalLifoDiskQueue, PickleFifoDiskQueue, PickleLifoDiskQueue
from scrapy.item import Item, Field
from scrapy.http import Request
from scrapy.loader import ItemLoader
import sys
import unittest

import six

from scrapy.item import ABCMeta, Item, ItemMeta, Field
from tests import mock


PY36_PLUS = (sys.version_info.major >= 3) and (sys.version_info.minor >= 6)

from os.path import join, abspath
from twisted.trial import unittest
from twisted.internet import defer
from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest
from scrapy.utils.python import to_native_str
from tests.test_commands import CommandTest

from __future__ import absolute_import
import os
import csv
import json
from io import BytesIO
import tempfile
import shutil
from six.moves.urllib.parse import urlparse

from zope.interface.verify import verifyObject
from twisted.trial import unittest
from twisted.internet import defer
from scrapy.crawler import CrawlerRunner
from scrapy.settings import Settings
from tests.mockserver import MockServer
from w3lib.url import path_to_file_uri

import scrapy
from scrapy.extensions.feedexport import (
    IFeedStorage, FileFeedStorage, FTPFeedStorage,
    S3FeedStorage, StdoutFeedStorage,
    BlockingFeedStorage)
from scrapy.utils.test import assert_aws_environ, get_s3_content_and_delete, get_crawler
from scrapy.utils.python import to_native_str

# -*- coding: utf-8 -*-

import unittest

from scrapy.downloadermiddlewares.redirect import RedirectMiddleware, MetaRefreshMiddleware
from scrapy.spiders import Spider
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response, HtmlResponse
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import cgi
import unittest
import re

import six
from six.moves import xmlrpc_client as xmlrpclib
from six.moves.urllib.parse import urlparse, parse_qs, unquote
if six.PY3:
    from urllib.parse import unquote_to_bytes

from scrapy.http import Request, FormRequest, XmlRpcRequest, Headers, HtmlResponse
from scrapy.utils.python import to_bytes, to_native_str

# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import logging
import unittest

from testfixtures import LogCapture
from twisted.python.failure import Failure

from scrapy.utils.log import (failure_to_exc_info, TopLevelFormatter,
                              LogCounterHandler, StreamLogger)
from scrapy.utils.test import get_crawler

import copy
import unittest
from collections import Mapping, MutableMapping

from scrapy.utils.datatypes import CaselessDict, SequenceExclude

__doctests__ = ['scrapy.utils.datatypes']
from testfixtures import LogCapture
from twisted.trial import unittest
from twisted.python.failure import Failure
from twisted.internet import defer, reactor
from pydispatch import dispatcher

from scrapy.utils.signal import send_catch_log, send_catch_log_deferred

import os
from shutil import rmtree
from tempfile import mkdtemp
import unittest
from scrapy.utils.template import render_templatefile


__doctests__ = ['scrapy.utils.template']

# -*- coding: utf-8 -*-
import unittest

import six
from w3lib.encoding import resolve_encoding

from scrapy.http import (Request, Response, TextResponse, HtmlResponse,
                         XmlResponse, Headers)
from scrapy.selector import Selector
from scrapy.utils.python import to_native_str
from scrapy.exceptions import NotSupported
from scrapy.link import Link
from tests import get_testdata

import re
import logging
from unittest import TestCase
from testfixtures import LogCapture

from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
from scrapy.exceptions import NotConfigured
from scrapy.downloadermiddlewares.cookies import CookiesMiddleware

import unittest
from twisted.internet import defer
from twisted.internet.error import TimeoutError, DNSLookupError, \
        ConnectionRefusedError, ConnectionDone, ConnectError, \
        ConnectionLost, TCPTimedOutError
from twisted.web.client import ResponseFailed

from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request, Response
from scrapy.utils.test import get_crawler

from six.moves.urllib.parse import urlparse
from unittest import TestCase

from scrapy.http import Request, Response
from scrapy.http.cookies import WrappedRequest, WrappedResponse

import unittest
from six.moves.urllib.parse import urlparse

from scrapy.http import Request
from scrapy.utils.httpobj import urlparse_cached
"""
from twisted.internet import defer
Tests borrowed from the twisted.web.client tests.
"""
import os
import six
import shutil

from twisted.trial import unittest
from twisted.web import server, static, util, resource
from twisted.internet import reactor, defer
from twisted.test.proto_helpers import StringTransport
from twisted.python.filepath import FilePath
from twisted.protocols.policies import WrappingFactory
from twisted.internet.defer import inlineCallbacks

from scrapy.core.downloader import webclient as client
from scrapy.http import Request, Headers
from scrapy.utils.python import to_bytes, to_unicode

import os
import sys
from functools import partial
from twisted.trial.unittest import TestCase, SkipTest

from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.crawler import Crawler
from scrapy.settings import Settings

spider = Spider('foo')

from unittest import TestCase

from scrapy.spidermiddlewares.depth import DepthMiddleware
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.test import get_crawler

from __future__ import print_function
from testfixtures import LogCapture
from twisted.trial import unittest
from twisted.python.failure import Failure
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks

from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.request import request_fingerprint
from scrapy.pipelines.media import MediaPipeline
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.signal import disconnect_all
from scrapy import signals

"""
Scrapy engine tests

This starts a testing web server (using twisted.server.Site) and then crawls it
with the Scrapy crawler.

To view the testing web server in a browser you can start it by running this
module with the ``runserver`` argument::

    python test_engine.py runserver
"""

from __future__ import print_function
import sys, os, re
from six.moves.urllib.parse import urlparse

from twisted.internet import reactor, defer
from twisted.web import server, static, util
from twisted.trial import unittest

from scrapy import signals
from scrapy.core.engine import ExecutionEngine
from scrapy.utils.test import get_crawler
from pydispatch import dispatcher
from tests import tests_datadir
from scrapy.spiders import Spider
from scrapy.item import Item, Field
from scrapy.linkextractors import LinkExtractor
from scrapy.http import Request
from scrapy.utils.signal import disconnect_all

import logging
import os
import tempfile
import warnings
import unittest

import scrapy
from scrapy.crawler import Crawler, CrawlerRunner, CrawlerProcess
from scrapy.settings import Settings, default_settings
from scrapy.spiderloader import SpiderLoader
from scrapy.utils.log import configure_logging, get_scrapy_root_handler
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.misc import load_object
from scrapy.extensions.throttle import AutoThrottle

from six.moves.urllib.parse import urlparse
from unittest import TestCase
import warnings

from scrapy.exceptions import NotConfigured
from scrapy.http import Response, Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.downloadermiddlewares.redirect import RedirectMiddleware
from scrapy.spidermiddlewares.referer import RefererMiddleware, \
    POLICY_NO_REFERRER, POLICY_NO_REFERRER_WHEN_DOWNGRADE, \
    POLICY_SAME_ORIGIN, POLICY_ORIGIN, POLICY_ORIGIN_WHEN_CROSS_ORIGIN, \
    POLICY_SCRAPY_DEFAULT, POLICY_UNSAFE_URL, \
    POLICY_STRICT_ORIGIN, POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN, \
    DefaultReferrerPolicy, \
    NoReferrerPolicy, NoReferrerWhenDowngradePolicy, \
    OriginWhenCrossOriginPolicy, OriginPolicy, \
    StrictOriginWhenCrossOriginPolicy, StrictOriginPolicy, \
    SameOriginPolicy, UnsafeUrlPolicy, ReferrerPolicy

# -*- coding: utf-8 -*-
import unittest
from scrapy.responsetypes import responsetypes

from scrapy.http import Response, TextResponse, XmlResponse, HtmlResponse, Headers
from unittest import TestCase

from scrapy.downloadermiddlewares.stats import DownloaderStats
from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import unittest

from scrapy.http import Request, FormRequest
from scrapy.spiders import Spider
from scrapy.utils.reqser import request_to_dict, request_from_dict

import unittest
import six

from scrapy.spiders import Spider
from scrapy.http import Request, Response
from scrapy.item import Item, Field
from scrapy.logformatter import LogFormatter

# -*- coding: utf-8 -*-
import os
import shutil

from testfixtures import LogCapture
from twisted.internet import defer
from twisted.trial.unittest import TestCase
from w3lib.url import add_or_replace_parameter

from scrapy.crawler import CrawlerRunner
from scrapy import signals
from tests.mockserver import MockServer
from tests.spiders import SimpleSpider

import six
import unittest
from scrapy.utils import trackref
from tests import mock

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import inspect
import unittest
import warnings
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.deprecate import create_deprecated_class, update_classpath

from tests import mock

# -*- coding: utf-8 -*-
import os
import six
from twisted.trial import unittest

from scrapy.utils.iterators import csviter, xmliter, _body_or_str, xmliter_lxml
from scrapy.http import XmlResponse, TextResponse, Response
from tests import get_testdata

FOOBAR_NL = u"foo" + os.linesep + u"bar"

# coding=utf-8

import unittest
from io import BytesIO
from email.charset import Charset

from scrapy.mail import MailSender
from unittest import TestCase
import six
import scrapy

from __future__ import print_function
import time
import tempfile
import shutil
import unittest
import email.utils
from contextlib import contextmanager
import pytest

from scrapy.http import Response, HtmlResponse, Request
from scrapy.spiders import Spider
from scrapy.settings import Settings
from scrapy.exceptions import IgnoreRequest
from scrapy.utils.test import get_crawler
from scrapy.downloadermiddlewares.httpcache import HttpCacheMiddleware

from unittest import TestCase

from six.moves.urllib.parse import urlparse

from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.spidermiddlewares.offsite import OffsiteMiddleware
from scrapy.utils.test import get_crawler
import unittest
import six
from functools import partial

from scrapy.loader import ItemLoader
from scrapy.loader.processors import Join, Identity, TakeFirst, \
    Compose, MapCompose, SelectJmes
from scrapy.item import Item, Field
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

# test itemsimport unittest

from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
import json
import os
import time

from threading import Thread
from libmproxy import controller, proxy
from netlib import http_auth
from testfixtures import LogCapture

from twisted.internet import defer
from twisted.trial.unittest import TestCase
from scrapy.utils.test import get_crawler
from scrapy.http import Request
from tests.spiders import SimpleSpider, SingleRequestSpider
from tests.mockserver import MockServer

from os.path import join

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest

from tests import tests_datadir

import os
import six
import contextlib
import shutil
try:
    from unittest import mock
except ImportError:
    import mock

from twisted.trial import unittest
from twisted.protocols.policies import WrappingFactory
from twisted.python.filepath import FilePath
from twisted.internet import reactor, defer, error
from twisted.web import server, static, util, resource
from twisted.web._newclient import ResponseFailed
from twisted.web.http import _DataLoss
from twisted.web.test.test_webclient import ForeverTakingResource, \
        NoLengthResource, HostHeaderResource, \
        PayloadResource
from twisted.cred import portal, checkers, credentials
from w3lib.url import path_to_file_uri

from scrapy.core.downloader.handlers import DownloadHandlers
from scrapy.core.downloader.handlers.datauri import DataURIDownloadHandler
from scrapy.core.downloader.handlers.file import FileDownloadHandler
from scrapy.core.downloader.handlers.http import HTTPDownloadHandler, HttpDownloadHandler
from scrapy.core.downloader.handlers.http10 import HTTP10DownloadHandler
from scrapy.core.downloader.handlers.http11 import HTTP11DownloadHandler
from scrapy.core.downloader.handlers.s3 import S3DownloadHandler

from scrapy.spiders import Spider
from scrapy.http import Headers, Request
from scrapy.http.response.text import TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.settings import Settings
from scrapy.utils.test import get_crawler, skip_if_no_boto
from scrapy.utils.python import to_bytes
from scrapy.exceptions import NotConfigured

from tests.mockserver import MockServer, ssl_context_factory, Echo
from tests.spiders import SingleRequestSpider
import re
import unittest

import pytest

from scrapy.http import HtmlResponse, XmlResponse
from scrapy.link import Link
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from tests import get_testdata


# a hack to skip base class tests in pytestfrom twisted.trial.unittest import TestCase
from twisted.python.failure import Failure

from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.core.downloader.middleware import DownloaderMiddlewareManager
from scrapy.utils.test import get_crawler
from scrapy.utils.python import to_bytes
from tests import mock

import warnings
import weakref
from twisted.trial import unittest
from scrapy.http import TextResponse, HtmlResponse, XmlResponse
from scrapy.selector import Selector
from scrapy.selector.lxmlsel import XmlXPathSelector, HtmlXPathSelector, XPathSelector
from lxml import etree

import unittest

from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector, DummyStatsCollector
from scrapy.utils.test import get_crawler

"""
Selector tests for cssselect backend
"""
import warnings
from twisted.trial import unittest
from scrapy.selector.csstranslator import (
    ScrapyHTMLTranslator,
    ScrapyGenericTranslator,
    ScrapyXPathExpr
)

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest


class MixinStrictOrigin(object):
    scenarii = [
        # TLS or non-TLS to TLS or non-TLS: referrer origin is sent but not for downgrades
        ('https://example.com/page.html',   'https://example.com/not-page.html',    b'https://example.com/'),
        ('https://example.com/page.html',   'https://scrapy.org',                   b'https://example.com/'),
        ('http://example.com/page.html',    'http://scrapy.org',                    b'http://example.com/'),

        # downgrade: send nothing
        ('https://example.com/page.html',   'http://scrapy.org',                    None),

        # upgrade: send origin
        ('http://example.com/page.html',    'https://scrapy.org',                   b'http://example.com/'),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'https://scrapy.org',       b'https://example.com/'),
        ('https://user:password@example.com/page.html', 'http://scrapy.org',        None),
    ]


class CaselessDictTest(unittest.TestCase):

    def test_init_dict(self):
        seq = {'red': 1, 'black': 3}
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_pair_sequence(self):
        seq = (('red', 1), ('black', 3))
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_mapping(self):
        class MyMapping(Mapping):
            def __init__(self, **kwargs):
                self._d = kwargs

            def __getitem__(self, key):
                return self._d[key]

            def __iter__(self):
                return iter(self._d)

            def __len__(self):
                return len(self._d)

        seq = MyMapping(red=1, black=3)
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_mutable_mapping(self):
        class MyMutableMapping(MutableMapping):
            def __init__(self, **kwargs):
                self._d = kwargs

            def __getitem__(self, key):
                return self._d[key]

            def __setitem__(self, key, value):
                self._d[key] = value

            def __delitem__(self, key):
                del self._d[key]

            def __iter__(self):
                return iter(self._d)

            def __len__(self):
                return len(self._d)

        seq = MyMutableMapping(red=1, black=3)
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_caseless(self):
        d = CaselessDict()
        d['key_Lower'] = 1
        self.assertEqual(d['KEy_loWer'], 1)
        self.assertEqual(d.get('KEy_loWer'), 1)

        d['KEY_LOWER'] = 3
        self.assertEqual(d['key_Lower'], 3)
        self.assertEqual(d.get('key_Lower'), 3)

    def test_delete(self):
        d = CaselessDict({'key_lower': 1})
        del d['key_LOWER']
        self.assertRaises(KeyError, d.__getitem__, 'key_LOWER')
        self.assertRaises(KeyError, d.__getitem__, 'key_lower')

    def test_getdefault(self):
        d = CaselessDict()
        self.assertEqual(d.get('c', 5), 5)
        d['c'] = 10
        self.assertEqual(d.get('c', 5), 10)

    def test_setdefault(self):
        d = CaselessDict({'a': 1, 'b': 2})

        r = d.setdefault('A', 5)
        self.assertEqual(r, 1)
        self.assertEqual(d['A'], 1)

        r = d.setdefault('c', 5)
        self.assertEqual(r, 5)
        self.assertEqual(d['C'], 5)

    def test_fromkeys(self):
        keys = ('a', 'b')

        d = CaselessDict.fromkeys(keys)
        self.assertEqual(d['A'], None)
        self.assertEqual(d['B'], None)

        d = CaselessDict.fromkeys(keys, 1)
        self.assertEqual(d['A'], 1)
        self.assertEqual(d['B'], 1)

        instance = CaselessDict()
        d = instance.fromkeys(keys)
        self.assertEqual(d['A'], None)
        self.assertEqual(d['B'], None)

        d = instance.fromkeys(keys, 1)
        self.assertEqual(d['A'], 1)
        self.assertEqual(d['B'], 1)

    def test_contains(self):
        d = CaselessDict()
        d['a'] = 1
        assert 'a' in d

    def test_pop(self):
        d = CaselessDict()
        d['a'] = 1
        self.assertEqual(d.pop('A'), 1)
        self.assertRaises(KeyError, d.pop, 'A')

    def test_normkey(self):
        class MyDict(CaselessDict):
            def normkey(self, key):
                return key.title()

        d = MyDict()
        d['key-one'] = 2
        self.assertEqual(list(d.keys()), ['Key-One'])

    def test_normvalue(self):
        class MyDict(CaselessDict):
            def normvalue(self, value):
                if value is not None:
                    return value + 1

        d = MyDict({'key': 1})
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d['key'] = 1
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d.setdefault('key', 1)
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d.update({'key': 1})
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict.fromkeys(('key',), 1)
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

    def test_copy(self):
        h1 = CaselessDict({'header1': 'value'})
        h2 = copy.copy(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(h1.get('header1'), h2.get('header1'))
        assert isinstance(h2, CaselessDict)


class M3(object):

    def process(self, response, request, spider):
        pass


class CustomItemExporterTest(unittest.TestCase):

    def test_exporter_custom_serializer(self):
        class CustomItemExporter(BaseItemExporter):
            def serialize_field(self, field, name, value):
                if name == 'age':
                    return str(int(value) + 1)
                else:
                    return super(CustomItemExporter, self).serialize_field(field, name, value)

        i = TestItem(name=u'John', age='22')
        ie = CustomItemExporter()

        self.assertEqual(ie.serialize_field(i.fields['name'], 'name', i['name']), 'John')
        self.assertEqual(ie.serialize_field(i.fields['age'], 'age', i['age']), '23')

        i2 = {'name': u'John', 'age': '22'}
        self.assertEqual(ie.serialize_field({}, 'name', i2['name']), 'John')
        self.assertEqual(ie.serialize_field({}, 'age', i2['age']), '23')


if __name__ == '__main__':
    unittest.main()

class ResponseUtilsTest(unittest.TestCase):
    dummy_response = TextResponse(url='http://example.org/', body=b'dummy_response')

    def test_response_httprepr(self):
        r1 = Response("http://www.example.com")
        self.assertEqual(response_httprepr(r1), b'HTTP/1.1 200 OK\r\n\r\n')

        r1 = Response("http://www.example.com", status=404, headers={"Content-type": "text/html"}, body=b"Some body")
        self.assertEqual(response_httprepr(r1), b'HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\n\r\nSome body')

        r1 = Response("http://www.example.com", status=6666, headers={"Content-type": "text/html"}, body=b"Some body")
        self.assertEqual(response_httprepr(r1), b'HTTP/1.1 6666 \r\nContent-Type: text/html\r\n\r\nSome body')

    def test_open_in_browser(self):
        url = "http:///www.example.com/some/page.html"
        body = b"<html> <head> <title>test page</title> </head> <body>test body</body> </html>"

        def browser_open(burl):
            path = urlparse(burl).path
            if not os.path.exists(path):
                path = burl.replace('file://', '')
            with open(path, "rb") as f:
                bbody = f.read()
            self.assertIn(b'<base href="' + to_bytes(url) + b'">', bbody)
            return True
        response = HtmlResponse(url, body=body)
        assert open_in_browser(response, _openfunc=browser_open), \
            "Browser not called"

        resp = Response(url, body=body)
        self.assertRaises(TypeError, open_in_browser, resp, debug=True)

    def test_get_meta_refresh(self):
        r1 = HtmlResponse("http://www.example.com", body=b"""
        <html>
        <head><title>Dummy</title><meta http-equiv="refresh" content="5;url=http://example.org/newpage" /></head>
        <body>blahablsdfsal&amp;</body>
        </html>""")
        r2 = HtmlResponse("http://www.example.com", body=b"""
        <html>
        <head><title>Dummy</title><noScript>
        <meta http-equiv="refresh" content="5;url=http://example.org/newpage" /></head>
        </noSCRIPT>
        <body>blahablsdfsal&amp;</body>
        </html>""")
        r3 = HtmlResponse("http://www.example.com", body=b"""
    <noscript><meta http-equiv="REFRESH" content="0;url=http://www.example.com/newpage</noscript>
    <script type="text/javascript">
    if(!checkCookies()){
        document.write('<meta http-equiv="REFRESH" content="0;url=http://www.example.com/newpage">');
    }
    </script>
        """)
        self.assertEqual(get_meta_refresh(r1), (5.0, 'http://example.org/newpage'))
        self.assertEqual(get_meta_refresh(r2), (None, None))
        self.assertEqual(get_meta_refresh(r3), (None, None))

    def test_get_base_url(self):
        resp = HtmlResponse("http://www.example.com", body=b"""
        <html>
        <head><base href="http://www.example.com/img/" target="_blank"></head>
        <body>blahablsdfsal&amp;</body>
        </html>""")
        self.assertEqual(get_base_url(resp), "http://www.example.com/img/")

        resp2 = HtmlResponse("http://www.example.com", body=b"""
        <html><body>blahablsdfsal&amp;</body></html>""")
        self.assertEqual(get_base_url(resp2), "http://www.example.com")

    def test_response_status_message(self):
        self.assertEqual(response_status_message(200), '200 OK')
        self.assertEqual(response_status_message(404), '404 Not Found')
        self.assertEqual(response_status_message(573), "573 Unknown Status")

class TestUrlLengthMiddleware(TestCase):

    def test_process_spider_output(self):
        res = Response('http://scrapytest.org')

        short_url_req = Request('http://scrapytest.org/')
        long_url_req = Request('http://scrapytest.org/this_is_a_long_url')
        reqs = [short_url_req, long_url_req]

        mw = UrlLengthMiddleware(maxlength=25)
        spider = Spider('foo')
        out = list(mw.process_spider_output(res, reqs, spider))
        self.assertEqual(out, [short_url_req])


class MediaDownloadSpider(SimpleSpider):
    name = 'mediadownload'

    def _process_url(self, url):
        return url

    def parse(self, response):
        self.logger.info(response.headers)
        self.logger.info(response.text)
        item = {
            self.media_key: [],
            self.media_urls_key: [
                self._process_url(response.urljoin(href))
                    for href in response.xpath('''
                        //table[thead/tr/th="Filename"]
                            /tbody//a/@href
                        ''').extract()],
        }
        yield item


class FilesystemStorageGzipTest(FilesystemStorageTest):

    def _get_settings(self, **new_settings):
        new_settings.setdefault('HTTPCACHE_GZIP', True)
        return super(FilesystemStorageTest, self)._get_settings(**new_settings)

class StdoutFeedStorageTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_store(self):
        out = BytesIO()
        storage = StdoutFeedStorage('stdout:', _stdout=out)
        file = storage.open(scrapy.Spider("default"))
        file.write(b"content")
        yield storage.store(file)
        self.assertEqual(out.getvalue(), b"content")


class ToBytesTest(unittest.TestCase):
    def test_converting_a_unicode_object_to_an_utf_8_encoded_string(self):
        self.assertEqual(to_bytes(u'\xa3 49'), b'\xc2\xa3 49')

    def test_converting_a_unicode_object_to_a_latin_1_encoded_string(self):
        self.assertEqual(to_bytes(u'\xa3 49', 'latin-1'), b'\xa3 49')

    def test_converting_a_regular_bytes_to_bytes_should_return_the_same_object(self):
        self.assertEqual(to_bytes(b'lel\xf1e'), b'lel\xf1e')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        self.assertRaises(TypeError, to_bytes, unittest)

    def test_errors_argument(self):
        self.assertEqual(
            to_bytes(u'a\ufffdb', 'latin-1', errors='replace'),
            b'a?b'
        )


class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass


class TestLoader(ItemLoader):
    default_item_class = TestItem
    name_out = staticmethod(_test_procesor)

def nonserializable_object_test(self):
    try:
        pickle.dumps(lambda x: x)
    except Exception:
        # Trigger Twisted bug #7989
        import twisted.persisted.styles  # NOQA
        q = self.queue()
        self.assertRaises(ValueError, q.push, lambda x: x)
    else:
        # Use a different unpickleable object
        class A(object): pass
        a = A()
        a.__reduce__ = a.__reduce_ex__ = None
        q = self.queue()
        self.assertRaises(ValueError, q.push, a)

class WrappedResponseTest(TestCase):

    def setUp(self):
        self.response = Response("http://www.example.com/page.html",
                                 headers={"Content-TYpe": "text/html"})
        self.wrapped = WrappedResponse(self.response)

    def test_info(self):
        self.assertIs(self.wrapped.info(), self.wrapped)

    def test_getheaders(self):
        self.assertEqual(self.wrapped.getheaders('content-type'), ['text/html'])

    def test_get_all(self):
        # get_all result must be native string
        self.assertEqual(self.wrapped.get_all('content-type'), ['text/html'])

class Foo(trackref.object_ref):
    pass


class FileFeedStorageTest(unittest.TestCase):

    def test_store_file_uri(self):
        path = os.path.abspath(self.mktemp())
        uri = path_to_file_uri(path)
        return self._assert_stores(FileFeedStorage(uri), path)

    def test_store_file_uri_makedirs(self):
        path = os.path.abspath(self.mktemp())
        path = os.path.join(path, 'more', 'paths', 'file.txt')
        uri = path_to_file_uri(path)
        return self._assert_stores(FileFeedStorage(uri), path)

    def test_store_direct_path(self):
        path = os.path.abspath(self.mktemp())
        return self._assert_stores(FileFeedStorage(path), path)

    def test_store_direct_path_relative(self):
        path = self.mktemp()
        return self._assert_stores(FileFeedStorage(path), path)

    def test_interface(self):
        path = self.mktemp()
        st = FileFeedStorage(path)
        verifyObject(IFeedStorage, st)

    @defer.inlineCallbacks
    def _assert_stores(self, storage, path):
        spider = scrapy.Spider("default")
        file = storage.open(spider)
        file.write(b"content")
        yield storage.store(file)
        self.assertTrue(os.path.exists(path))
        try:
            with open(path, 'rb') as fp:
                self.assertEqual(fp.read(), b"content")
        finally:
            os.unlink(path)


class GuessSchemeTest(unittest.TestCase):
    pass

def create_guess_scheme_t(args):
    def do_expected(self):
        url = guess_scheme(args[0])
        assert url.startswith(args[1]), \
            'Wrong scheme guessed: for `%s` got `%s`, expected `%s...`' % (
                args[0], url, args[1])
    return do_expected

def create_skipped_scheme_t(args):
    def do_expected(self):
        raise unittest.SkipTest(args[2])
        url = guess_scheme(args[0])
        assert url.startswith(args[1])
    return do_expected

for k, args in enumerate ([
            ('/index',                              'file://'),
            ('/index.html',                         'file://'),
            ('./index.html',                        'file://'),
            ('../index.html',                       'file://'),
            ('../../index.html',                    'file://'),
            ('./data/index.html',                   'file://'),
            ('.hidden/data/index.html',             'file://'),
            ('/home/user/www/index.html',           'file://'),
            ('//home/user/www/index.html',          'file://'),
            ('file:///home/user/www/index.html',    'file://'),

            ('index.html',                          'http://'),
            ('example.com',                         'http://'),
            ('www.example.com',                     'http://'),
            ('www.example.com/index.html',          'http://'),
            ('http://example.com',                  'http://'),
            ('http://example.com/index.html',       'http://'),
            ('localhost',                           'http://'),
            ('localhost/index.html',                'http://'),

            # some corner cases (default to http://)
            ('/',                                   'http://'),
            ('.../test',                            'http://'),

        ], start=1):
    t_method = create_guess_scheme_t(args)
    t_method.__name__ = 'test_uri_%03d' % k
    setattr (GuessSchemeTest, t_method.__name__, t_method)

# TODO: the following tests do not pass with current implementation
for k, args in enumerate ([
            ('C:\absolute\path\to\a\file.html',     'file://',
             'Windows filepath are not supported for scrapy shell'),
        ], start=1):
    t_method = create_skipped_scheme_t(args)
    t_method.__name__ = 'test_uri_skipped_%03d' % k
    setattr (GuessSchemeTest, t_method.__name__, t_method)


class ChunkSize2PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 2

class TestRequestMetaSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SAME_ORIGIN}


class TestPolicyHeaderPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    resp_headers = {'Referrer-Policy': POLICY_UNSAFE_URL.upper()}

class DeprecatedHttpTestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HttpDownloadHandler


class TestSettingsSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}


class DataURITestCase(unittest.TestCase):

    def setUp(self):
        self.download_handler = DataURIDownloadHandler(Settings())
        self.download_request = self.download_handler.download_request
        self.spider = Spider('foo')

    def test_response_attrs(self):
        uri = "data:,A%20brief%20note"

        def _test(response):
            self.assertEqual(response.url, uri)
            self.assertFalse(response.headers)

        request = Request(uri)
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype_encoding(self):
        def _test(response):
            self.assertEqual(response.text, 'A brief note')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "US-ASCII")

        request = Request("data:,A%20brief%20note")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "iso-8859-7")

        request = Request("data:;charset=iso-8859-7,%be%d3%be")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_text_charset(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(response.body, b'\xbe\xd3\xbe')
            self.assertEqual(response.encoding, "iso-8859-7")

        request = Request("data:text/plain;charset=iso-8859-7,%be%d3%be")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_mediatype_parameters(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "utf-8")

        request = Request('data:text/plain;foo=%22foo;bar%5C%22%22;'
                          'charset=utf-8;bar=%22foo;%5C%22 foo ;/,%22'
                          ',%CE%8E%CE%A3%CE%8E')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_base64(self):
        def _test(response):
            self.assertEqual(response.text, 'Hello, world.')

        request = Request('data:text/plain;base64,SGVsbG8sIHdvcmxkLg%3D%3D')
        return self.download_request(request, self.spider).addCallback(_test)

class TestRequestMetaOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN_WHEN_CROSS_ORIGIN}


class XMLFeedSpiderTest(SpiderTest):

    spider_class = XMLFeedSpider

    def test_register_namespace(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns:x="http://www.google.com/schemas/sitemap/0.84"
                xmlns:y="http://www.example.com/schemas/extras/1.0">
        <url><x:loc>http://www.example.com/Special-Offers.html</loc><y:updated>2009-08-16</updated><other value="bar" y:custom="fuu"/></url>
        <url><loc>http://www.example.com/</loc><y:updated>2009-08-16</updated><other value="foo"/></url>
        </urlset>"""
        response = XmlResponse(url='http://example.com/sitemap.xml', body=body)

        class _XMLSpider(self.spider_class):
            itertag = 'url'
            namespaces = (
                ('a', 'http://www.google.com/schemas/sitemap/0.84'),
                ('b', 'http://www.example.com/schemas/extras/1.0'),
            )

            def parse_node(self, response, selector):
                yield {
                    'loc': selector.xpath('a:loc/text()').extract(),
                    'updated': selector.xpath('b:updated/text()').extract(),
                    'other': selector.xpath('other/@value').extract(),
                    'custom': selector.xpath('other/@b:custom').extract(),
                }

        for iterator in ('iternodes', 'xml'):
            spider = _XMLSpider('example', iterator=iterator)
            output = list(spider.parse(response))
            self.assertEqual(len(output), 2, iterator)
            self.assertEqual(output, [
                {'loc': [u'http://www.example.com/Special-Offers.html'],
                 'updated': [u'2009-08-16'],
                 'custom': [u'fuu'],
                 'other': [u'bar']},
                {'loc': [],
                 'updated': [u'2009-08-16'],
                 'other': [u'foo'],
                 'custom': []},
            ], iterator)


class TestSettingsOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}


class NameItem(Item):
    name = Field()


class TestRequestMetaPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass


class DeprecatedHttpTestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HttpDownloadHandler


class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass

