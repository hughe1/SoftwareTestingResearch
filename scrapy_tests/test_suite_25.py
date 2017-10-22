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


class UrlparseTestCase(unittest.TestCase):

    def test_s3_url(self):
        p = urlparse('s3://bucket/key/name?param=value')
        self.assertEqual(p.scheme, 's3')
        self.assertEqual(p.hostname, 'bucket')
        self.assertEqual(p.path, '/key/name')
        self.assertEqual(p.query, 'param=value')

class TestSettingsUnsafeUrl(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.UnsafeUrlPolicy'}


class ToplevelTestCase(TestCase):

    def test_version(self):
        self.assertIs(type(scrapy.__version__), six.text_type)

    def test_version_info(self):
        self.assertIs(type(scrapy.version_info), tuple)

    def test_request_shortcut(self):
        from scrapy.http import Request, FormRequest
        self.assertIs(scrapy.Request, Request)
        self.assertIs(scrapy.FormRequest, FormRequest)

    def test_spider_shortcut(self):
        from scrapy.spiders import Spider
        self.assertIs(scrapy.Spider, Spider)

    def test_selector_shortcut(self):
        from scrapy.selector import Selector
        self.assertIs(scrapy.Selector, Selector)

    def test_item_shortcut(self):
        from scrapy.item import Item, Field
        self.assertIs(scrapy.Item, Item)
        self.assertIs(scrapy.Field, Field)

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


class TestHttpErrorMiddlewareHandleAll(TestCase):

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpErrorMiddleware(Settings({'HTTPERROR_ALLOW_ALL': True}))
        self.req = Request('http://scrapytest.org')
        self.res200, self.res404, self.res402 = _responses(self.req, [200, 404, 402])

    def test_process_spider_input(self):
        self.assertEqual(None,
                self.mw.process_spider_input(self.res200, self.spider))
        self.assertEqual(None,
                self.mw.process_spider_input(self.res404, self.spider))

    def test_meta_overrides_settings(self):
        request = Request('http://scrapytest.org',
                              meta={'handle_httpstatus_list': [404]})
        res404 = self.res404.copy()
        res404.request = request
        res402 = self.res402.copy()
        res402.request = request

        self.assertEqual(None,
            self.mw.process_spider_input(res404, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, res402, self.spider)


class DefaultsTest(ManagerTestCase):
    """Tests default behavior with default settings"""

    def test_request_response(self):
        req = Request('http://example.com/index.html')
        resp = Response(req.url, status=200)
        ret = self._download(req, resp)
        self.assertTrue(isinstance(ret, Response), "Non-response returned")

    def test_3xx_and_invalid_gzipped_body_must_redirect(self):
        """Regression test for a failure when redirecting a compressed
        request.

        This happens when httpcompression middleware is executed before redirect
        middleware and attempts to decompress a non-compressed body.
        In particular when some website returns a 30x response with header
        'Content-Encoding: gzip' giving as result the error below:

            exceptions.IOError: Not a gzipped file

        """
        req = Request('http://example.com')
        body = b'<p>You are being redirected</p>'
        resp = Response(req.url, status=302, body=body, headers={
            'Content-Length': str(len(body)),
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
            'Location': 'http://example.com/login',
        })
        ret = self._download(request=req, response=resp)
        self.assertTrue(isinstance(ret, Request),
                        "Not redirected: {0!r}".format(ret))
        self.assertEqual(to_bytes(ret.url), resp.headers['Location'],
                         "Not redirected to location header")

    def test_200_and_invalid_gzipped_body_must_fail(self):
        req = Request('http://example.com')
        body = b'<p>You are being redirected</p>'
        resp = Response(req.url, status=200, body=body, headers={
            'Content-Length': str(len(body)),
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
            'Location': 'http://example.com/login',
        })
        self.assertRaises(IOError, self._download, request=req, response=resp)


class TestReferrerOnRedirectOriginWhenCrossOrigin(TestReferrerOnRedirect):
    """
    Origin When Cross-Origin policy sends the full URL as "Referer",
    unless the target's origin is different (different domain, different protocol)
    in which case only the origin is sent.

    Redirections to a different origin should strip the "Referer"
    to the parent origin.
    """
    settings = {'REFERRER_POLICY': POLICY_ORIGIN_WHEN_CROSS_ORIGIN}
    scenarii = [
        (   'http://scrapytest.org/101',      # origin
            'http://scrapytest.org/102',      # target + redirection
            (
                # redirections: code, URL
                (301, 'http://scrapytest.org/103'),
                (301, 'http://scrapytest.org/104'),
            ),
            b'http://scrapytest.org/101', # expected initial referer
            b'http://scrapytest.org/101', # expected referer for the redirection request
        ),
        (   'https://scrapytest.org/201',
            'https://scrapytest.org/202',
            (
                # redirecting to non-secure URL: send origin
                (301, 'http://scrapytest.org/203'),
            ),
            b'https://scrapytest.org/201',
            b'https://scrapytest.org/',
        ),
        (   'https://scrapytest.org/301',
            'https://scrapytest.org/302',
            (
                # redirecting to non-secure URL (different domain): send origin
                (301, 'http://example.com/303'),
            ),
            b'https://scrapytest.org/301',
            b'https://scrapytest.org/',
        ),
        (   'http://scrapy.org/401',
            'http://example.com/402',
            (
                (301, 'http://scrapytest.org/403'),
            ),
            b'http://scrapy.org/',
            b'http://scrapy.org/',
        ),
        (   'https://scrapy.org/501',
            'https://example.com/502',
            (
                # all different domains: send origin
                (301, 'https://google.com/503'),
                (301, 'https://facebook.com/504'),
            ),
            b'https://scrapy.org/',
            b'https://scrapy.org/',
        ),
        (   'https://scrapytest.org/301',
            'http://scrapytest.org/302',                # TLS to non-TLS: send origin
            (
                (301, 'https://scrapytest.org/303'),    # TLS URL again: send origin (also)
            ),
            b'https://scrapytest.org/',
            b'https://scrapytest.org/',
        ),
    ]


class TestItemLoader(NameItemLoader):
    name_in = MapCompose(lambda v: v.title())


class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass


class UtilsCsvTestCase(unittest.TestCase):
    sample_feeds_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'sample_data', 'feeds')
    sample_feed_path = os.path.join(sample_feeds_dir, 'feed-sample3.csv')
    sample_feed2_path = os.path.join(sample_feeds_dir, 'feed-sample4.csv')
    sample_feed3_path = os.path.join(sample_feeds_dir, 'feed-sample5.csv')

    def test_csviter_defaults(self):
        body = get_testdata('feeds', 'feed-sample3.csv')
        response = TextResponse(url="http://example.com/", body=body)
        csv = csviter(response)

        result = [row for row in csv]
        self.assertEqual(result,
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

        # explicit type check cuz' we no like stinkin' autocasting! yarrr
        for result_row in result:
            self.assertTrue(all((isinstance(k, six.text_type) for k in result_row.keys())))
            self.assertTrue(all((isinstance(v, six.text_type) for v in result_row.values())))

    def test_csviter_delimiter(self):
        body = get_testdata('feeds', 'feed-sample3.csv').replace(b',', b'\t')
        response = TextResponse(url="http://example.com/", body=body)
        csv = csviter(response, delimiter='\t')

        self.assertEqual([row for row in csv],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

    def test_csviter_quotechar(self):
        body1 = get_testdata('feeds', 'feed-sample6.csv')
        body2 = get_testdata('feeds', 'feed-sample6.csv').replace(b',', b'|')

        response1 = TextResponse(url="http://example.com/", body=body1)
        csv1 = csviter(response1, quotechar="'")

        self.assertEqual([row for row in csv1],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

        response2 = TextResponse(url="http://example.com/", body=body2)
        csv2 = csviter(response2, delimiter="|", quotechar="'")

        self.assertEqual([row for row in csv2],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

    def test_csviter_wrong_quotechar(self):
        body = get_testdata('feeds', 'feed-sample6.csv')
        response = TextResponse(url="http://example.com/", body=body)
        csv = csviter(response)

        self.assertEqual([row for row in csv],
                         [{u"'id'": u"1",   u"'name'": u"'alpha'",   u"'value'": u"'foobar'"},
                          {u"'id'": u"2",   u"'name'": u"'unicode'", u"'value'": u"'\xfan\xedc\xf3d\xe9\u203d'"},
                          {u"'id'": u"'3'", u"'name'": u"'multi'",   u"'value'": u"'foo"},
                          {u"'id'": u"4",   u"'name'": u"'empty'",   u"'value'": u""}])

    def test_csviter_delimiter_binary_response_assume_utf8_encoding(self):
        body = get_testdata('feeds', 'feed-sample3.csv').replace(b',', b'\t')
        response = Response(url="http://example.com/", body=body)
        csv = csviter(response, delimiter='\t')

        self.assertEqual([row for row in csv],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

    def test_csviter_headers(self):
        sample = get_testdata('feeds', 'feed-sample3.csv').splitlines()
        headers, body = sample[0].split(b','), b'\n'.join(sample[1:])

        response = TextResponse(url="http://example.com/", body=body)
        csv = csviter(response, headers=[h.decode('utf-8') for h in headers])

        self.assertEqual([row for row in csv],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': u'foo\nbar'},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

    def test_csviter_falserow(self):
        body = get_testdata('feeds', 'feed-sample3.csv')
        body = b'\n'.join((body, b'a,b', b'a,b,c,d'))

        response = TextResponse(url="http://example.com/", body=body)
        csv = csviter(response)

        self.assertEqual([row for row in csv],
                         [{u'id': u'1', u'name': u'alpha',   u'value': u'foobar'},
                          {u'id': u'2', u'name': u'unicode', u'value': u'\xfan\xedc\xf3d\xe9\u203d'},
                          {u'id': u'3', u'name': u'multi',   u'value': FOOBAR_NL},
                          {u'id': u'4', u'name': u'empty',   u'value': u''}])

    def test_csviter_exception(self):
        body = get_testdata('feeds', 'feed-sample3.csv')

        response = TextResponse(url="http://example.com/", body=body)
        iter = csviter(response)
        next(iter)
        next(iter)
        next(iter)
        next(iter)

        self.assertRaises(StopIteration, next, iter)

    def test_csviter_encoding(self):
        body1 = get_testdata('feeds', 'feed-sample4.csv')
        body2 = get_testdata('feeds', 'feed-sample5.csv')

        response = TextResponse(url="http://example.com/", body=body1, encoding='latin1')
        csv = csviter(response)
        self.assertEqual([row for row in csv],
            [{u'id': u'1', u'name': u'latin1', u'value': u'test'},
             {u'id': u'2', u'name': u'something', u'value': u'\xf1\xe1\xe9\xf3'}])

        response = TextResponse(url="http://example.com/", body=body2, encoding='cp852')
        csv = csviter(response)
        self.assertEqual([row for row in csv],
            [{u'id': u'1', u'name': u'cp852', u'value': u'test'},
             {u'id': u'2', u'name': u'something', u'value': u'\u255a\u2569\u2569\u2569\u2550\u2550\u2557'}])


class ChunkSize2MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 2

class BaseCrawlerTest(unittest.TestCase):

    def assertOptionIsDefault(self, settings, key):
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings[key], getattr(default_settings, key))


class PprintItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return PprintItemExporter(self.output, **kwargs)

    def _check_output(self):
        self._assert_expected_item(eval(self.output.getvalue()))


class SgmlLinkExtractorTestCase(Base.LinkExtractorTestCase):
    extractor_cls = SgmlLinkExtractor
    escapes_whitespace = True

    def test_deny_extensions(self):
        html = """<a href="page.html">asd</a> and <a href="photo.jpg">"""
        response = HtmlResponse("http://example.org/", body=html)
        lx = SgmlLinkExtractor(deny_extensions="jpg")
        self.assertEqual(lx.extract_links(response), [
            Link(url='http://example.org/page.html', text=u'asd'),
        ])

    def test_attrs_sgml(self):
        html = """<html><area href="sample1.html"></area>
        <a ref="sample2.html">sample text 2</a></html>"""
        response = HtmlResponse("http://example.com/index.html", body=html)
        lx = SgmlLinkExtractor(attrs="href")
        self.assertEqual(lx.extract_links(response), [
            Link(url='http://example.com/sample1.html', text=u''),
        ])

    def test_link_nofollow(self):
        html = """
        <a href="page.html?action=print" rel="nofollow">Printer-friendly page</a>
        <a href="about.html">About us</a>
        <a href="http://google.com/something" rel="external nofollow">Something</a>
        """
        response = HtmlResponse("http://example.org/page.html", body=html)
        lx = SgmlLinkExtractor()
        self.assertEqual([link for link in lx.extract_links(response)], [
            Link(url='http://example.org/page.html?action=print', text=u'Printer-friendly page', nofollow=True),
            Link(url='http://example.org/about.html', text=u'About us', nofollow=False),
            Link(url='http://google.com/something', text=u'Something', nofollow=True),
        ])


class DummyDH(object):

    def __init__(self, crawler):
        pass


class TestHttpErrorMiddlewareSettings(TestCase):
    """Similar test, but with settings"""

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpErrorMiddleware(Settings({'HTTPERROR_ALLOWED_CODES': (402,)}))
        self.req = Request('http://scrapytest.org')
        self.res200, self.res404, self.res402 = _responses(self.req, [200, 404, 402])

    def test_process_spider_input(self):
        self.assertEqual(None,
                self.mw.process_spider_input(self.res200, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, self.res404, self.spider)
        self.assertEqual(None,
                self.mw.process_spider_input(self.res402, self.spider))

    def test_meta_overrides_settings(self):
        request = Request('http://scrapytest.org',
                              meta={'handle_httpstatus_list': [404]})
        res404 = self.res404.copy()
        res404.request = request
        res402 = self.res402.copy()
        res402.request = request

        self.assertEqual(None,
            self.mw.process_spider_input(res404, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, res402, self.spider)

    def test_spider_override_settings(self):
        self.spider.handle_httpstatus_list = [404]
        self.assertEqual(None,
            self.mw.process_spider_input(self.res404, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, self.res402, self.spider)


class LoadTestCase(unittest.TestCase):

    def test_enabled_handler(self):
        handlers = {'scheme': 'tests.test_downloader_handlers.DummyDH'}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertIn('scheme', dh._handlers)
        self.assertNotIn('scheme', dh._notconfigured)

    def test_not_configured_handler(self):
        handlers = {'scheme': 'tests.test_downloader_handlers.OffDH'}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)

    def test_disabled_handler(self):
        handlers = {'scheme': None}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertNotIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)


class ResponseTypesTest(unittest.TestCase):

    def test_from_filename(self):
        mappings = [
            ('data.bin', Response),
            ('file.txt', TextResponse),
            ('file.xml.gz', Response),
            ('file.xml', XmlResponse),
            ('file.html', HtmlResponse),
            ('file.unknownext', Response),
        ]
        for source, cls in mappings:
            retcls = responsetypes.from_filename(source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_from_content_disposition(self):
        mappings = [
            (b'attachment; filename="data.xml"', XmlResponse),
            (b'attachment; filename=data.xml', XmlResponse),
            (u'attachment;filename=data£.tar.gz'.encode('utf-8'), Response),
            (u'attachment;filename=dataµ.tar.gz'.encode('latin-1'), Response),
            (u'attachment;filename=data高.doc'.encode('gbk'), Response),
            (u'attachment;filename=دورهdata.html'.encode('cp720'), HtmlResponse),
            (u'attachment;filename=日本語版Wikipedia.xml'.encode('iso2022_jp'), XmlResponse),

        ]
        for source, cls in mappings:
            retcls = responsetypes.from_content_disposition(source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_from_content_type(self):
        mappings = [
            ('text/html; charset=UTF-8', HtmlResponse),
            ('text/xml; charset=UTF-8', XmlResponse),
            ('application/xhtml+xml; charset=UTF-8', HtmlResponse),
            ('application/vnd.wap.xhtml+xml; charset=utf-8', HtmlResponse),
            ('application/xml; charset=UTF-8', XmlResponse),
            ('application/octet-stream', Response),
            ('application/x-json; encoding=UTF8;charset=UTF-8', TextResponse),
            ('application/json-amazonui-streaming;charset=UTF-8', TextResponse),
        ]
        for source, cls in mappings:
            retcls = responsetypes.from_content_type(source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_from_body(self):
        mappings = [
            (b'\x03\x02\xdf\xdd\x23', Response),
            (b'Some plain text\ndata with tabs\t and null bytes\0', TextResponse),
            (b'<html><head><title>Hello</title></head>', HtmlResponse),
            (b'<?xml version="1.0" encoding="utf-8"', XmlResponse),
        ]
        for source, cls in mappings:
            retcls = responsetypes.from_body(source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_from_headers(self):
        mappings = [
            ({'Content-Type': ['text/html; charset=utf-8']}, HtmlResponse),
            ({'Content-Type': ['application/octet-stream'], 'Content-Disposition': ['attachment; filename=data.txt']}, TextResponse),
            ({'Content-Type': ['text/html; charset=utf-8'], 'Content-Encoding': ['gzip']}, Response),
        ]
        for source, cls in mappings:
            source = Headers(source)
            retcls = responsetypes.from_headers(source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_from_args(self):
        # TODO: add more tests that check precedence between the different arguments
        mappings = [
            ({'url': 'http://www.example.com/data.csv'}, TextResponse),
            # headers takes precedence over url
            ({'headers': Headers({'Content-Type': ['text/html; charset=utf-8']}), 'url': 'http://www.example.com/item/'}, HtmlResponse),
            ({'headers': Headers({'Content-Disposition': ['attachment; filename="data.xml.gz"']}), 'url': 'http://www.example.com/page/'}, Response),


        ]
        for source, cls in mappings:
            retcls = responsetypes.from_args(**source)
            assert retcls is cls, "%s ==> %s != %s" % (source, retcls, cls)

    def test_custom_mime_types_loaded(self):
        # check that mime.types files shipped with scrapy are loaded
        self.assertEqual(responsetypes.mimetypes.guess_type('x.scrapytest')[0], 'x-scrapy/test')

if __name__ == "__main__":
    unittest.main()

class TestPolicyHeaderPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER.swapcase()}

class FTPFeedStorageTest(unittest.TestCase):

    def test_store(self):
        uri = os.environ.get('FEEDTEST_FTP_URI')
        path = os.environ.get('FEEDTEST_FTP_PATH')
        if not (uri and path):
            raise unittest.SkipTest("No FTP server available for testing")
        st = FTPFeedStorage(uri)
        verifyObject(IFeedStorage, st)
        return self._assert_stores(st, path)

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
            # again, to check s3 objects are overwritten
            yield storage.store(BytesIO(b"new content"))
            with open(path, 'rb') as fp:
                self.assertEqual(fp.read(), b"new content")
        finally:
            os.unlink(path)


class M3(object):

    def process(self, response, request, spider):
        pass


class Bar(trackref.object_ref):
    pass


class TestRequestMetaOrigin(MixinOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN}


class CrawlerProcessTest(BaseCrawlerTest):
    def test_crawler_process_accepts_dict(self):
        runner = CrawlerProcess({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_process_accepts_None(self):
        runner = CrawlerProcess()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

class SequenceExcludeTest(unittest.TestCase):

    def test_list(self):
        seq = [1, 2, 3]
        d = SequenceExclude(seq)
        self.assertIn(0, d)
        self.assertIn(4, d)
        self.assertNotIn(2, d)

    def test_range(self):
        seq = range(10, 20)
        d = SequenceExclude(seq)
        self.assertIn(5, d)
        self.assertIn(20, d)
        self.assertNotIn(15, d)

    def test_six_range(self):
        import six.moves
        seq = six.moves.range(10**3, 10**6)
        d = SequenceExclude(seq)
        self.assertIn(10**2, d)
        self.assertIn(10**7, d)
        self.assertNotIn(10**4, d)

    def test_range_step(self):
        seq = range(10, 20, 3)
        d = SequenceExclude(seq)
        are_not_in = [v for v in range(10, 20, 3) if v in d]
        self.assertEqual([], are_not_in)

        are_not_in = [v for v in range(10, 20) if v in d]
        self.assertEqual([11, 12, 14, 15, 17, 18], are_not_in)

    def test_string_seq(self):
        seq = "cde"
        d = SequenceExclude(seq)
        chars = "".join(v for v in "abcdefg" if v in d)
        self.assertEqual("abfg", chars)

    def test_stringset_seq(self):
        seq = set("cde")
        d = SequenceExclude(seq)
        chars = "".join(v for v in "abcdefg" if v in d)
        self.assertEqual("abfg", chars)

    def test_set(self):
        """Anything that is not in the supplied sequence will evaluate as 'in' the container."""
        seq = set([-3, "test", 1.1])
        d = SequenceExclude(seq)
        self.assertIn(0, d)
        self.assertIn("foo", d)
        self.assertIn(3.14, d)
        self.assertIn(set("bar"), d)

        # supplied sequence is a set, so checking for list (non)inclusion fails
        self.assertRaises(TypeError, (0, 1, 2) in d)
        self.assertRaises(TypeError, d.__contains__, ['a', 'b', 'c'])

        for v in [-3, "test", 1.1]:
            self.assertNotIn(v, d)

if __name__ == "__main__":
    unittest.main()


class Https10TestCase(Http10TestCase):
    scheme = 'https'


class FilesPipelineTestCaseCustomSettings(unittest.TestCase):
    default_cls_settings = {
        "EXPIRES": 90,
        "FILES_URLS_FIELD": "file_urls",
        "FILES_RESULT_FIELD": "files"
    }
    file_cls_attr_settings_map = {
        ("EXPIRES", "FILES_EXPIRES", "expires"),
        ("FILES_URLS_FIELD", "FILES_URLS_FIELD", "files_urls_field"),
        ("FILES_RESULT_FIELD", "FILES_RESULT_FIELD", "files_result_field")
    }

    def setUp(self):
        self.tempdir = mkdtemp()

    def tearDown(self):
        rmtree(self.tempdir)

    def _generate_fake_settings(self, prefix=None):

        def random_string():
            return "".join([chr(random.randint(97, 123)) for _ in range(10)])

        settings = {
            "FILES_EXPIRES": random.randint(100, 1000),
            "FILES_URLS_FIELD": random_string(),
            "FILES_RESULT_FIELD": random_string(),
            "FILES_STORE": self.tempdir
        }
        if not prefix:
            return settings

        return {prefix.upper() + "_" + k if k != "FILES_STORE" else k: v for k, v in settings.items()}

    def _generate_fake_pipeline(self):

        class UserDefinedFilePipeline(FilesPipeline):
            EXPIRES = 1001
            FILES_URLS_FIELD = "alfa"
            FILES_RESULT_FIELD = "beta"

        return UserDefinedFilePipeline

    def test_different_settings_for_different_instances(self):
        """
        If there are different instances with different settings they should keep
        different settings.
        """
        custom_settings = self._generate_fake_settings()
        another_pipeline = FilesPipeline.from_settings(Settings(custom_settings))
        one_pipeline = FilesPipeline(self.tempdir)
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            default_value = self.default_cls_settings[pipe_attr]
            self.assertEqual(getattr(one_pipeline, pipe_attr), default_value)
            custom_value = custom_settings[settings_attr]
            self.assertNotEqual(default_value, custom_value)
            self.assertEqual(getattr(another_pipeline, pipe_ins_attr), custom_value)

    def test_subclass_attributes_preserved_if_no_settings(self):
        """
        If subclasses override class attributes and there are no special settings those values should be kept.
        """
        pipe_cls = self._generate_fake_pipeline()
        pipe = pipe_cls.from_settings(Settings({"FILES_STORE": self.tempdir}))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            custom_value = getattr(pipe, pipe_ins_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_attr])
            self.assertEqual(getattr(pipe, pipe_ins_attr), getattr(pipe, pipe_attr))

    def test_subclass_attrs_preserved_custom_settings(self):
        """
        If file settings are defined but they are not defined for subclass
        settings should be preserved.
        """
        pipeline_cls = self._generate_fake_pipeline()
        settings = self._generate_fake_settings()
        pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            value = getattr(pipeline, pipe_ins_attr)
            setting_value = settings.get(settings_attr)
            self.assertNotEqual(value, self.default_cls_settings[pipe_attr])
            self.assertEqual(value, setting_value)

    def test_no_custom_settings_for_subclasses(self):
        """
        If there are no settings for subclass and no subclass attributes, pipeline should use
        attributes of base class.
        """
        class UserDefinedFilesPipeline(FilesPipeline):
            pass

        user_pipeline = UserDefinedFilesPipeline.from_settings(Settings({"FILES_STORE": self.tempdir}))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = self.default_cls_settings.get(pipe_attr.upper())
            self.assertEqual(getattr(user_pipeline, pipe_ins_attr), custom_value)

    def test_custom_settings_for_subclasses(self):
        """
        If there are custom settings for subclass and NO class attributes, pipeline should use custom
        settings.
        """
        class UserDefinedFilesPipeline(FilesPipeline):
            pass

        prefix = UserDefinedFilesPipeline.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = UserDefinedFilesPipeline.from_settings(Settings(settings))
        for pipe_attr, settings_attr, pipe_inst_attr in self.file_cls_attr_settings_map:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_attr])
            self.assertEqual(getattr(user_pipeline, pipe_inst_attr), custom_value)

    def test_custom_settings_and_class_attrs_for_subclasses(self):
        """
        If there are custom settings for subclass AND class attributes
        setting keys are preferred and override attributes.
        """
        pipeline_cls = self._generate_fake_pipeline()
        prefix = pipeline_cls.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_cls_attr, settings_attr, pipe_inst_attr  in self.file_cls_attr_settings_map:
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_cls_attr])
            self.assertEqual(getattr(user_pipeline, pipe_inst_attr), custom_value)

    def test_cls_attrs_with_DEFAULT_prefix(self):
        class UserDefinedFilesPipeline(FilesPipeline):
            DEFAULT_FILES_RESULT_FIELD = "this"
            DEFAULT_FILES_URLS_FIELD = "that"

        pipeline = UserDefinedFilesPipeline.from_settings(Settings({"FILES_STORE": self.tempdir}))
        self.assertEqual(pipeline.files_result_field, "this")
        self.assertEqual(pipeline.files_urls_field, "that")


    def test_user_defined_subclass_default_key_names(self):
        """Test situation when user defines subclass of FilesPipeline,
        but uses attribute names for default pipeline (without prefixing
        them with pipeline class name).
        """
        settings = self._generate_fake_settings()

        class UserPipe(FilesPipeline):
            pass

        pipeline_cls = UserPipe.from_settings(Settings(settings))

        for pipe_attr, settings_attr, pipe_inst_attr in self.file_cls_attr_settings_map:
            expected_value = settings.get(settings_attr)
            self.assertEqual(getattr(pipeline_cls, pipe_inst_attr),
                             expected_value)


class BaseSgmlLinkExtractorTestCase(unittest.TestCase):
    # XXX: should we move some of these tests to base link extractor tests?

    def test_basic(self):
        html = """<html><head><title>Page title<title>
        <body><p><a href="item/12.html">Item 12</a></p>
        <p><a href="/about.html">About us</a></p>
        <img src="/logo.png" alt="Company logo (not a link)" />
        <p><a href="../othercat.html">Other category</a></p>
        <p><a href="/">&gt;&gt;</a></p>
        <p><a href="/" /></p>
        </body></html>"""
        response = HtmlResponse("http://example.org/somepage/index.html", body=html)

        lx = BaseSgmlLinkExtractor()  # default: tag=a, attr=href
        self.assertEqual(lx.extract_links(response),
                         [Link(url='http://example.org/somepage/item/12.html', text='Item 12'),
                          Link(url='http://example.org/about.html', text='About us'),
                          Link(url='http://example.org/othercat.html', text='Other category'),
                          Link(url='http://example.org/', text='>>'),
                          Link(url='http://example.org/', text='')])

    def test_base_url(self):
        html = """<html><head><title>Page title<title><base href="http://otherdomain.com/base/" />
        <body><p><a href="item/12.html">Item 12</a></p>
        </body></html>"""
        response = HtmlResponse("http://example.org/somepage/index.html", body=html)

        lx = BaseSgmlLinkExtractor()  # default: tag=a, attr=href
        self.assertEqual(lx.extract_links(response),
                         [Link(url='http://otherdomain.com/base/item/12.html', text='Item 12')])

        # base url is an absolute path and relative to host
        html = """<html><head><title>Page title<title><base href="/" />
        <body><p><a href="item/12.html">Item 12</a></p></body></html>"""
        response = HtmlResponse("https://example.org/somepage/index.html", body=html)
        self.assertEqual(lx.extract_links(response),
                         [Link(url='https://example.org/item/12.html', text='Item 12')])

        # base url has no scheme
        html = """<html><head><title>Page title<title><base href="//noschemedomain.com/path/to/" />
        <body><p><a href="item/12.html">Item 12</a></p></body></html>"""
        response = HtmlResponse("https://example.org/somepage/index.html", body=html)
        self.assertEqual(lx.extract_links(response),
                         [Link(url='https://noschemedomain.com/path/to/item/12.html', text='Item 12')])

    def test_link_text_wrong_encoding(self):
        html = """<body><p><a href="item/12.html">Wrong: \xed</a></p></body></html>"""
        response = HtmlResponse("http://www.example.com", body=html, encoding='utf-8')
        lx = BaseSgmlLinkExtractor()
        self.assertEqual(lx.extract_links(response), [
            Link(url='http://www.example.com/item/12.html', text=u'Wrong: \ufffd'),
        ])

    def test_extraction_encoding(self):
        body = get_testdata('link_extractor', 'linkextractor_noenc.html')
        response_utf8 = HtmlResponse(url='http://example.com/utf8', body=body, headers={'Content-Type': ['text/html; charset=utf-8']})
        response_noenc = HtmlResponse(url='http://example.com/noenc', body=body)
        body = get_testdata('link_extractor', 'linkextractor_latin1.html')
        response_latin1 = HtmlResponse(url='http://example.com/latin1', body=body)

        lx = BaseSgmlLinkExtractor()
        self.assertEqual(lx.extract_links(response_utf8), [
            Link(url='http://example.com/sample_%C3%B1.html', text=''),
            Link(url='http://example.com/sample_%E2%82%AC.html', text='sample \xe2\x82\xac text'.decode('utf-8')),
        ])

        self.assertEqual(lx.extract_links(response_noenc), [
            Link(url='http://example.com/sample_%C3%B1.html', text=''),
            Link(url='http://example.com/sample_%E2%82%AC.html', text='sample \xe2\x82\xac text'.decode('utf-8')),
        ])

        # document encoding does not affect URL path component, only query part
        # >>> u'sample_ñ.html'.encode('utf8')
        # b'sample_\xc3\xb1.html'
        # >>> u"sample_á.html".encode('utf8')
        # b'sample_\xc3\xa1.html'
        # >>> u"sample_ö.html".encode('utf8')
        # b'sample_\xc3\xb6.html'
        # >>> u"£32".encode('latin1')
        # b'\xa332'
        # >>> u"µ".encode('latin1')
        # b'\xb5'
        self.assertEqual(lx.extract_links(response_latin1), [
            Link(url='http://example.com/sample_%C3%B1.html', text=''),
            Link(url='http://example.com/sample_%C3%A1.html', text='sample \xe1 text'.decode('latin1')),
            Link(url='http://example.com/sample_%C3%B6.html?price=%A332&%B5=unit', text=''),
        ])

    def test_matches(self):
        url1 = 'http://lotsofstuff.com/stuff1/index'
        url2 = 'http://evenmorestuff.com/uglystuff/index'

        lx = BaseSgmlLinkExtractor()
        self.assertEqual(lx.matches(url1), True)
        self.assertEqual(lx.matches(url2), True)


class StartprojectTest(ProjectTest):

    def test_startproject(self):
        self.assertEqual(0, self.call('startproject', self.project_name))

        assert exists(join(self.proj_path, 'scrapy.cfg'))
        assert exists(join(self.proj_path, 'testproject'))
        assert exists(join(self.proj_mod_path, '__init__.py'))
        assert exists(join(self.proj_mod_path, 'items.py'))
        assert exists(join(self.proj_mod_path, 'pipelines.py'))
        assert exists(join(self.proj_mod_path, 'settings.py'))
        assert exists(join(self.proj_mod_path, 'spiders', '__init__.py'))

        self.assertEqual(1, self.call('startproject', self.project_name))
        self.assertEqual(1, self.call('startproject', 'wrong---project---name'))
        self.assertEqual(1, self.call('startproject', 'sys'))

    def test_startproject_with_project_dir(self):
        project_dir = mkdtemp()
        self.assertEqual(0, self.call('startproject', self.project_name, project_dir))

        assert exists(join(abspath(project_dir), 'scrapy.cfg'))
        assert exists(join(abspath(project_dir), 'testproject'))
        assert exists(join(join(abspath(project_dir), self.project_name), '__init__.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'items.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'pipelines.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'settings.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'spiders', '__init__.py'))

        self.assertEqual(0, self.call('startproject', self.project_name, project_dir + '2'))

        self.assertEqual(1, self.call('startproject', self.project_name, project_dir))
        self.assertEqual(1, self.call('startproject', self.project_name + '2', project_dir))
        self.assertEqual(1, self.call('startproject', 'wrong---project---name'))
        self.assertEqual(1, self.call('startproject', 'sys'))
        self.assertEqual(2, self.call('startproject'))
        self.assertEqual(2, self.call('startproject', self.project_name, project_dir, 'another_params'))


class DownloadTimeoutMiddlewareTest(unittest.TestCase):

    def get_request_spider_mw(self, settings=None):
        crawler = get_crawler(Spider, settings)
        spider = crawler._create_spider('foo')
        request = Request('http://scrapytest.org/')
        return request, spider, DownloadTimeoutMiddleware.from_crawler(crawler)

    def test_default_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 180)

    def test_string_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw({'DOWNLOAD_TIMEOUT': '20.1'})
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 20.1)

    def test_spider_has_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        spider.download_timeout = 2
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 2)

    def test_request_has_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        spider.download_timeout = 2
        mw.spider_opened(spider)
        req.meta['download_timeout'] = 1
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 1)
