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


class HeadersTest(unittest.TestCase):

    def assertSortedEqual(self, first, second, msg=None):
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_basics(self):
        h = Headers({'Content-Type': 'text/html', 'Content-Length': 1234})
        assert h['Content-Type']
        assert h['Content-Length']

        self.assertRaises(KeyError, h.__getitem__, 'Accept')
        self.assertEqual(h.get('Accept'), None)
        self.assertEqual(h.getlist('Accept'), [])

        self.assertEqual(h.get('Accept', '*/*'), b'*/*')
        self.assertEqual(h.getlist('Accept', '*/*'), [b'*/*'])
        self.assertEqual(h.getlist('Accept', ['text/html', 'images/jpeg']),
                         [b'text/html', b'images/jpeg'])

    def test_single_value(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        self.assertEqual(h['Content-Type'], b'text/html')
        self.assertEqual(h.get('Content-Type'), b'text/html')
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])

    def test_multivalue(self):
        h = Headers()
        h['X-Forwarded-For'] = hlist = ['ip1', 'ip2']
        self.assertEqual(h['X-Forwarded-For'], b'ip2')
        self.assertEqual(h.get('X-Forwarded-For'), b'ip2')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])
        assert h.getlist('X-Forwarded-For') is not hlist

    def test_encode_utf8(self):
        h = Headers({u'key': u'\xa3'}, encoding='utf-8')
        key, val = dict(h).popitem()
        assert isinstance(key, bytes), key
        assert isinstance(val[0], bytes), val[0]
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_encode_latin1(self):
        h = Headers({u'key': u'\xa3'}, encoding='latin1')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xa3')

    def test_encode_multiple(self):
        h = Headers({u'key': [u'\xa3']}, encoding='utf-8')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_delete_and_contains(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        assert 'Content-Type' in h
        del h['Content-Type']
        assert 'Content-Type' not in h

    def test_setdefault(self):
        h = Headers()
        hlist = ['ip1', 'ip2']
        olist = h.setdefault('X-Forwarded-For', hlist)
        assert h.getlist('X-Forwarded-For') is not hlist
        assert h.getlist('X-Forwarded-For') is olist

        h = Headers()
        olist = h.setdefault('X-Forwarded-For', 'ip1')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1'])
        assert h.getlist('X-Forwarded-For') is olist

    def test_iterables(self):
        idict = {'Content-Type': 'text/html', 'X-Forwarded-For': ['ip1', 'ip2']}

        h = Headers(idict)
        self.assertDictEqual(dict(h),
                             {b'Content-Type': [b'text/html'],
                              b'X-Forwarded-For': [b'ip1', b'ip2']})
        self.assertSortedEqual(h.keys(),
                               [b'X-Forwarded-For', b'Content-Type'])
        self.assertSortedEqual(h.items(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.iteritems(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.values(), [b'ip2', b'text/html'])

    def test_update(self):
        h = Headers()
        h.update({'Content-Type': 'text/html',
                  'X-Forwarded-For': ['ip1', 'ip2']})
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])

    def test_copy(self):
        h1 = Headers({'header1': ['value1', 'value2']})
        h2 = copy.copy(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(h1.getlist('header1'), h2.getlist('header1'))
        assert h1.getlist('header1') is not h2.getlist('header1')
        assert isinstance(h2, Headers)

    def test_appendlist(self):
        h1 = Headers({'header1': 'value1'})
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

        h1 = Headers()
        h1.appendlist('header1', 'value1')
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

    def test_setlist(self):
        h1 = Headers({'header1': 'value1'})
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        h1.setlist('header1', [b'value2', b'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value2', b'value3'])

    def test_setlistdefault(self):
        h1 = Headers({'header1': 'value1'})
        h1.setlistdefault('header1', ['value2', 'value3'])
        h1.setlistdefault('header2', ['value2', 'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        self.assertEqual(h1.getlist('header2'), [b'value2', b'value3'])

    def test_none_value(self):
        h1 = Headers()
        h1['foo'] = 'bar'
        h1['foo'] = None
        h1.setdefault('foo', 'bar')
        self.assertEqual(h1.get('foo'), None)
        self.assertEqual(h1.getlist('foo'), [])

    def test_int_value(self):
        h1 = Headers({'hey': 5})
        h1['foo'] = 1
        h1.setdefault('bar', 2)
        h1.setlist('buz', [1, 'dos', 3])
        self.assertEqual(h1.getlist('foo'), [b'1'])
        self.assertEqual(h1.getlist('bar'), [b'2'])
        self.assertEqual(h1.getlist('buz'), [b'1', b'dos', b'3'])
        self.assertEqual(h1.getlist('hey'), [b'5'])

    def test_invalid_value(self):
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers, {'foo': object()})
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().__setitem__, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setdefault, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setlist, 'foo', [object()])

class MyBadCrawlSpider(CrawlSpider):
    '''Spider which doesn't define a parse_item callback while using it in a rule.'''
    name = 'badcrawl{0}'

    rules = (
        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),
    )

    def parse(self, response):
        return [scrapy.Item(), dict(foo='bar')]
""".format(self.spider_name))

        fname = abspath(join(self.proj_mod_path, 'pipelines.py'))
        with open(fname, 'w') as f:
            f.write("""
import logging

class S3TestCase(unittest.TestCase):
    download_handler_cls = S3DownloadHandler

    # test use same example keys than amazon developer guide
    # http://s3.amazonaws.com/awsdocs/S3/20060301/s3-dg-20060301.pdf
    # and the tests described here are the examples from that manual

    AWS_ACCESS_KEY_ID = '0PN5J17HBGZHT7JJ3X82'
    AWS_SECRET_ACCESS_KEY = 'uV3F3YluFJax1cknvbcGwgjvx4QpvB+leU8dUj2o'

    def setUp(self):
        skip_if_no_boto()
        s3reqh = S3DownloadHandler(Settings(), self.AWS_ACCESS_KEY_ID,
                self.AWS_SECRET_ACCESS_KEY,
                httpdownloadhandler=HttpDownloadHandlerMock)
        self.download_request = s3reqh.download_request
        self.spider = Spider('foo')

    @contextlib.contextmanager
    def _mocked_date(self, date):
        try:
            import botocore.auth
        except ImportError:
            yield
        else:
            # We need to mock botocore.auth.formatdate, because otherwise
            # botocore overrides Date header with current date and time
            # and Authorization header is different each time
            with mock.patch('botocore.auth.formatdate') as mock_formatdate:
                mock_formatdate.return_value = date
                yield

    def test_extra_kw(self):
        try:
            S3DownloadHandler(Settings(), extra_kw=True)
        except Exception as e:
            self.assertIsInstance(e, (TypeError, NotConfigured))
        else:
            assert False

    def test_request_signing1(self):
        # gets an object from the johnsmith bucket.
        date ='Tue, 27 Mar 2007 19:36:42 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:xXjDGYUmKxnwqr5KXNPGldn5LbA=')

    def test_request_signing2(self):
        # puts an object into the johnsmith bucket.
        date = 'Tue, 27 Mar 2007 21:15:45 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', method='PUT', headers={
            'Content-Type': 'image/jpeg',
            'Date': date,
            'Content-Length': '94328',
            })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:hcicpDDvL9SsO6AkvxqmIWkmOuQ=')

    def test_request_signing3(self):
        # lists the content of the johnsmith bucket.
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request('s3://johnsmith/?prefix=photos&max-keys=50&marker=puppy', \
                method='GET', headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Date': date,
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:jsRt/rhG+Vtp88HrYL706QhE4w4=')

    def test_request_signing4(self):
        # fetches the access control policy sub-resource for the 'johnsmith' bucket.
        date = 'Tue, 27 Mar 2007 19:44:46 +0000'
        req = Request('s3://johnsmith/?acl',
            method='GET', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:thdUi9VAkzhkniLj96JIrOPGi0g=')

    def test_request_signing5(self):
        try: import botocore
        except ImportError: pass
        else:
            raise unittest.SkipTest(
                'botocore does not support overriding date with x-amz-date')
        # deletes an object from the 'johnsmith' bucket using the
        # path-style and Date alternative.
        date = 'Tue, 27 Mar 2007 21:20:27 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', \
                method='DELETE', headers={
                    'Date': date,
                    'x-amz-date': 'Tue, 27 Mar 2007 21:20:26 +0000',
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        # botocore does not override Date with x-amz-date
        self.assertEqual(httpreq.headers['Authorization'],
                b'AWS 0PN5J17HBGZHT7JJ3X82:k3nL7gH3+PadhTEVn5Ip83xlYzk=')

    def test_request_signing6(self):
        # uploads an object to a CNAME style virtual hosted bucket with metadata.
        date = 'Tue, 27 Mar 2007 21:06:08 +0000'
        req = Request('s3://static.johnsmith.net:8080/db-backup.dat.gz', \
                method='PUT', headers={
                    'User-Agent': 'curl/7.15.5',
                    'Host': 'static.johnsmith.net:8080',
                    'Date': date,
                    'x-amz-acl': 'public-read',
                    'content-type': 'application/x-download',
                    'Content-MD5': '4gJE4saaMU4BqNR0kLY+lw==',
                    'X-Amz-Meta-ReviewedBy': 'joe@johnsmith.net,jane@johnsmith.net',
                    'X-Amz-Meta-FileChecksum': '0x02661779',
                    'X-Amz-Meta-ChecksumAlgorithm': 'crc32',
                    'Content-Disposition': 'attachment; filename=database.dat',
                    'Content-Encoding': 'gzip',
                    'Content-Length': '5913339',
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:C0FlOtU8Ylb9KDTpZqYkZPX91iI=')

    def test_request_signing7(self):
        # ensure that spaces are quoted properly before signing
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request(
            ("s3://johnsmith/photos/my puppy.jpg"
             "?response-content-disposition=my puppy.jpg"),
            method='GET',
            headers={'Date': date},
            )
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(
            httpreq.headers['Authorization'],
            b'AWS 0PN5J17HBGZHT7JJ3X82:+CfvG8EZ3YccOrRVMXNaK2eKZmM=')


class UtilsRequestTest(unittest.TestCase):

    def test_request_fingerprint(self):
        r1 = Request("http://www.example.com/query?id=111&cat=222")
        r2 = Request("http://www.example.com/query?cat=222&id=111")
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r1))
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2))

        r1 = Request('http://www.example.com/hnnoticiaj1.aspx?78132,199')
        r2 = Request('http://www.example.com/hnnoticiaj1.aspx?78160,199')
        self.assertNotEqual(request_fingerprint(r1), request_fingerprint(r2))

        # make sure caching is working
        self.assertEqual(request_fingerprint(r1), _fingerprint_cache[r1][None])

        r1 = Request("http://www.example.com/members/offers.html")
        r2 = Request("http://www.example.com/members/offers.html")
        r2.headers['SESSIONID'] = b"somehash"
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2))

        r1 = Request("http://www.example.com/")
        r2 = Request("http://www.example.com/")
        r2.headers['Accept-Language'] = b'en'
        r3 = Request("http://www.example.com/")
        r3.headers['Accept-Language'] = b'en'
        r3.headers['SESSIONID'] = b"somehash"

        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2), request_fingerprint(r3))

        self.assertEqual(request_fingerprint(r1),
                         request_fingerprint(r1, include_headers=['Accept-Language']))

        self.assertNotEqual(request_fingerprint(r1),
                         request_fingerprint(r2, include_headers=['Accept-Language']))

        self.assertEqual(request_fingerprint(r3, include_headers=['accept-language', 'sessionid']),
                         request_fingerprint(r3, include_headers=['SESSIONID', 'Accept-Language']))

        r1 = Request("http://www.example.com")
        r2 = Request("http://www.example.com", method='POST')
        r3 = Request("http://www.example.com", method='POST', body=b'request body')

        self.assertNotEqual(request_fingerprint(r1), request_fingerprint(r2))
        self.assertNotEqual(request_fingerprint(r2), request_fingerprint(r3))

        # cached fingerprint must be cleared on request copy
        r1 = Request("http://www.example.com")
        fp1 = request_fingerprint(r1)
        r2 = r1.replace(url="http://www.example.com/other")
        fp2 = request_fingerprint(r2)
        self.assertNotEqual(fp1, fp2)

    def test_request_authenticate(self):
        r = Request("http://www.example.com")
        request_authenticate(r, 'someuser', 'somepass')
        self.assertEqual(r.headers['Authorization'], b'Basic c29tZXVzZXI6c29tZXBhc3M=')

    def test_request_httprepr(self):
        r1 = Request("http://www.example.com")
        self.assertEqual(request_httprepr(r1), b'GET / HTTP/1.1\r\nHost: www.example.com\r\n\r\n')

        r1 = Request("http://www.example.com/some/page.html?arg=1")
        self.assertEqual(request_httprepr(r1), b'GET /some/page.html?arg=1 HTTP/1.1\r\nHost: www.example.com\r\n\r\n')

        r1 = Request("http://www.example.com", method='POST', headers={"Content-type": b"text/html"}, body=b"Some body")
        self.assertEqual(request_httprepr(r1), b'POST / HTTP/1.1\r\nHost: www.example.com\r\nContent-Type: text/html\r\n\r\nSome body')

    def test_request_httprepr_for_non_http_request(self):
        # the representation is not important but it must not fail.
        request_httprepr(Request("file:///tmp/foo.txt"))
        request_httprepr(Request("ftp://localhost/tmp/foo.txt"))

if __name__ == "__main__":
    unittest.main()

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

class MixinNoReferrerWhenDowngrade(object):
    scenarii = [
        # TLS to TLS: send non-empty referrer
        ('https://example.com/page.html',       'https://not.example.com/', b'https://example.com/page.html'),
        ('https://example.com/page.html',       'https://scrapy.org/',      b'https://example.com/page.html'),
        ('https://example.com:443/page.html',   'https://scrapy.org/',      b'https://example.com/page.html'),
        ('https://example.com:444/page.html',   'https://scrapy.org/',      b'https://example.com:444/page.html'),
        ('ftps://example.com/urls.zip',         'https://scrapy.org/',      b'ftps://example.com/urls.zip'),

        # TLS to non-TLS: do not send referrer
        ('https://example.com/page.html',       'http://not.example.com/',  None),
        ('https://example.com/page.html',       'http://scrapy.org/',       None),
        ('ftps://example.com/urls.zip',         'http://scrapy.org/',       None),

        # non-TLS to TLS or non-TLS: send referrer
        ('http://example.com/page.html',        'https://not.example.com/', b'http://example.com/page.html'),
        ('http://example.com/page.html',        'https://scrapy.org/',      b'http://example.com/page.html'),
        ('http://example.com:8080/page.html',   'https://scrapy.org/',      b'http://example.com:8080/page.html'),
        ('http://example.com:80/page.html',     'http://not.example.com/',  b'http://example.com/page.html'),
        ('http://example.com/page.html',        'http://scrapy.org/',       b'http://example.com/page.html'),
        ('http://example.com:443/page.html',    'http://scrapy.org/',       b'http://example.com:443/page.html'),
        ('ftp://example.com/urls.zip',          'http://scrapy.org/',       b'ftp://example.com/urls.zip'),
        ('ftp://example.com/urls.zip',          'https://scrapy.org/',      b'ftp://example.com/urls.zip'),

        # test for user/password stripping
        ('http://user:password@example.com/page.html', 'https://not.example.com/', b'http://example.com/page.html'),
    ]


class TestRequestMetaPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class SpiderSettingsTestCase(unittest.TestCase):
    def test_spider_custom_settings(self):
        class MySpider(scrapy.Spider):
            name = 'spider'
            custom_settings = {
                'AUTOTHROTTLE_ENABLED': True
            }

        crawler = Crawler(MySpider, {})
        enabled_exts = [e.__class__ for e in crawler.extensions.middlewares]
        self.assertIn(AutoThrottle, enabled_exts)


class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass


class HttpCompressionTest(TestCase):

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpCompressionMiddleware()

    def _getresponse(self, coding):
        if coding not in FORMAT:
            raise ValueError()

        samplefile, contentencoding = FORMAT[coding]

        with open(join(SAMPLEDIR, samplefile), 'rb') as sample:
            body = sample.read()

        headers = {
                'Server': 'Yaws/1.49 Yet Another Web Server',
                'Date': 'Sun, 08 Mar 2009 00:41:03 GMT',
                'Content-Length': len(body),
                'Content-Type': 'text/html',
                'Content-Encoding': contentencoding,
                }

        response = Response('http://scrapytest.org/', body=body, headers=headers)
        response.request = Request('http://scrapytest.org', headers={'Accept-Encoding': 'gzip,deflate'})
        return response

    def test_process_request(self):
        request = Request('http://scrapytest.org')
        assert 'Accept-Encoding' not in request.headers
        self.mw.process_request(request, self.spider)
        self.assertEqual(request.headers.get('Accept-Encoding'),
                         b','.join(ACCEPTED_ENCODINGS))

    def test_process_response_gzip(self):
        response = self._getresponse('gzip')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'gzip')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_br(self):
        try:
            import brotli
        except ImportError:
            raise SkipTest("no brotli")
        response = self._getresponse('br')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'br')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b"<!DOCTYPE")
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_rawdeflate(self):
        response = self._getresponse('rawdeflate')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_zlibdelate(self):
        response = self._getresponse('zlibdeflate')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_plain(self):
        response = Response('http://scrapytest.org', body=b'<!DOCTYPE...')
        request = Request('http://scrapytest.org')

        assert not response.headers.get('Content-Encoding')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is response
        assert newresponse.body.startswith(b'<!DOCTYPE')

    def test_multipleencodings(self):
        response = self._getresponse('gzip')
        response.headers['Content-Encoding'] = ['uuencode', 'gzip']
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        self.assertEqual(newresponse.headers.getlist('Content-Encoding'), [b'uuencode'])

    def test_process_response_encoding_inside_body(self):
        headers = {
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
        }
        f = BytesIO()
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = Response("http;//www.example.com/", headers=headers, body=f.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_force_recalculate_encoding(self):
        headers = {
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
        }
        f = BytesIO()
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = HtmlResponse("http;//www.example.com/page.html", headers=headers, body=f.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_no_content_type_header(self):
        headers = {
            'Content-Encoding': 'identity',
        }
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        respcls = responsetypes.from_args(url="http://www.example.com/index", headers=headers, body=plainbody)
        response = respcls("http://www.example.com/index", headers=headers, body=plainbody)
        request = Request("http://www.example.com/index")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, respcls)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_gzipped_contenttype(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzip_app_octetstream_contenttype(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/octet-stream'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzip_binary_octetstream_contenttype(self):
        response = self._getresponse('x-gzip')
        response.headers['Content-Type'] = 'binary/octet-stream'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzipped_gzip_file(self):
        """Test that a gzip Content-Encoded .gz file is gunzipped
        only once by the middleware, leaving gunzipping of the file
        to upper layers.
        """
        headers = {
            'Content-Type': 'application/gzip',
            'Content-Encoding': 'gzip',
        }
        # build a gzipped file (here, a sitemap)
        f = BytesIO()
        plainbody = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url>
    <loc>http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url>
    <loc>http://www.example.com/Special-Offers.html</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
        gz_file = GzipFile(fileobj=f, mode='wb')
        gz_file.write(plainbody)
        gz_file.close()

        # build a gzipped response body containing this gzipped file
        r = BytesIO()
        gz_resp = GzipFile(fileobj=r, mode='wb')
        gz_resp.write(f.getvalue())
        gz_resp.close()

        response = Response("http;//www.example.com/", headers=headers, body=r.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertEqual(gunzip(newresponse.body), plainbody)

    def test_process_response_head_request_no_decode_required(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request
        request.method = 'HEAD'
        response = response.replace(body = None)
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIs(newresponse, response)
        self.assertEqual(response.body, b'')

class XmlRpcRequestTest(RequestTest):

    request_class = XmlRpcRequest
    default_method = 'POST'
    default_headers = {b'Content-Type': [b'text/xml']}

    def _test_request(self, **kwargs):
        r = self.request_class('http://scrapytest.org/rpc2', **kwargs)
        self.assertEqual(r.headers[b'Content-Type'], b'text/xml')
        self.assertEqual(r.body,
                         to_bytes(xmlrpclib.dumps(**kwargs),
                                  encoding=kwargs.get('encoding', 'utf-8')))
        self.assertEqual(r.method, 'POST')
        self.assertEqual(r.encoding, kwargs.get('encoding', 'utf-8'))
        self.assertTrue(r.dont_filter, True)

    def test_xmlrpc_dumps(self):
        self._test_request(params=('value',))
        self._test_request(params=('username', 'password'), methodname='login')
        self._test_request(params=('response', ), methodresponse='login')
        self._test_request(params=(u'pas£',), encoding='utf-8')
        self._test_request(params=(None,), allow_none=1)
        self.assertRaises(TypeError, self._test_request)
        self.assertRaises(TypeError, self._test_request, params=(None,))

    def test_latin1(self):
        self._test_request(params=(u'pas£',), encoding='latin1')


if __name__ == "__main__":
    unittest.main()

class ParseUrlTestCase(unittest.TestCase):
    """Test URL parsing facility and defaults values."""

    def _parse(self, url):
        f = client.ScrapyHTTPClientFactory(Request(url))
        return (f.scheme, f.netloc, f.host, f.port, f.path)

    def testParse(self):
        lip = '127.0.0.1'
        tests = (
    ("http://127.0.0.1?c=v&c2=v2#fragment",     ('http', lip, lip, 80, '/?c=v&c2=v2')),
    ("http://127.0.0.1/?c=v&c2=v2#fragment",    ('http', lip, lip, 80, '/?c=v&c2=v2')),
    ("http://127.0.0.1/foo?c=v&c2=v2#frag",     ('http', lip, lip, 80, '/foo?c=v&c2=v2')),
    ("http://127.0.0.1:100?c=v&c2=v2#fragment", ('http', lip+':100', lip, 100, '/?c=v&c2=v2')),
    ("http://127.0.0.1:100/?c=v&c2=v2#frag",    ('http', lip+':100', lip, 100, '/?c=v&c2=v2')),
    ("http://127.0.0.1:100/foo?c=v&c2=v2#frag", ('http', lip+':100', lip, 100, '/foo?c=v&c2=v2')),

    ("http://127.0.0.1",              ('http', lip, lip, 80, '/')),
    ("http://127.0.0.1/",             ('http', lip, lip, 80, '/')),
    ("http://127.0.0.1/foo",          ('http', lip, lip, 80, '/foo')),
    ("http://127.0.0.1?param=value",  ('http', lip, lip, 80, '/?param=value')),
    ("http://127.0.0.1/?param=value", ('http', lip, lip, 80, '/?param=value')),
    ("http://127.0.0.1:12345/foo",    ('http', lip+':12345', lip, 12345, '/foo')),
    ("http://spam:12345/foo",         ('http', 'spam:12345', 'spam', 12345, '/foo')),
    ("http://spam.test.org/foo",      ('http', 'spam.test.org', 'spam.test.org', 80, '/foo')),

    ("https://127.0.0.1/foo",         ('https', lip, lip, 443, '/foo')),
    ("https://127.0.0.1/?param=value", ('https', lip, lip, 443, '/?param=value')),
    ("https://127.0.0.1:12345/",      ('https', lip+':12345', lip, 12345, '/')),

    ("http://scrapytest.org/foo ",    ('http', 'scrapytest.org', 'scrapytest.org', 80, '/foo')),
    ("http://egg:7890 ",              ('http', 'egg:7890', 'egg', 7890, '/')),
    )

        for url, test in tests:
            test = tuple(
                to_bytes(x) if not isinstance(x, int) else x for x in test)
            self.assertEqual(client._parse(url), test, url)

    def test_externalUnicodeInterference(self):
        """
        L{client._parse} should return C{str} for the scheme, host, and path
        elements of its return tuple, even when passed an URL which has
        previously been passed to L{urlparse} as a C{unicode} string.
        """
        if not six.PY2:
            raise unittest.SkipTest(
                "Applies only to Py2, as urls can be ONLY unicode on Py3")
        badInput = u'http://example.com/path'
        goodInput = badInput.encode('ascii')
        self._parse(badInput)  # cache badInput in urlparse_cached
        scheme, netloc, host, port, path = self._parse(goodInput)
        self.assertTrue(isinstance(scheme, str))
        self.assertTrue(isinstance(netloc, str))
        self.assertTrue(isinstance(host, str))
        self.assertTrue(isinstance(path, str))
        self.assertTrue(isinstance(port, int))



class HeadersTest(unittest.TestCase):

    def assertSortedEqual(self, first, second, msg=None):
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_basics(self):
        h = Headers({'Content-Type': 'text/html', 'Content-Length': 1234})
        assert h['Content-Type']
        assert h['Content-Length']

        self.assertRaises(KeyError, h.__getitem__, 'Accept')
        self.assertEqual(h.get('Accept'), None)
        self.assertEqual(h.getlist('Accept'), [])

        self.assertEqual(h.get('Accept', '*/*'), b'*/*')
        self.assertEqual(h.getlist('Accept', '*/*'), [b'*/*'])
        self.assertEqual(h.getlist('Accept', ['text/html', 'images/jpeg']),
                         [b'text/html', b'images/jpeg'])

    def test_single_value(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        self.assertEqual(h['Content-Type'], b'text/html')
        self.assertEqual(h.get('Content-Type'), b'text/html')
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])

    def test_multivalue(self):
        h = Headers()
        h['X-Forwarded-For'] = hlist = ['ip1', 'ip2']
        self.assertEqual(h['X-Forwarded-For'], b'ip2')
        self.assertEqual(h.get('X-Forwarded-For'), b'ip2')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])
        assert h.getlist('X-Forwarded-For') is not hlist

    def test_encode_utf8(self):
        h = Headers({u'key': u'\xa3'}, encoding='utf-8')
        key, val = dict(h).popitem()
        assert isinstance(key, bytes), key
        assert isinstance(val[0], bytes), val[0]
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_encode_latin1(self):
        h = Headers({u'key': u'\xa3'}, encoding='latin1')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xa3')

    def test_encode_multiple(self):
        h = Headers({u'key': [u'\xa3']}, encoding='utf-8')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_delete_and_contains(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        assert 'Content-Type' in h
        del h['Content-Type']
        assert 'Content-Type' not in h

    def test_setdefault(self):
        h = Headers()
        hlist = ['ip1', 'ip2']
        olist = h.setdefault('X-Forwarded-For', hlist)
        assert h.getlist('X-Forwarded-For') is not hlist
        assert h.getlist('X-Forwarded-For') is olist

        h = Headers()
        olist = h.setdefault('X-Forwarded-For', 'ip1')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1'])
        assert h.getlist('X-Forwarded-For') is olist

    def test_iterables(self):
        idict = {'Content-Type': 'text/html', 'X-Forwarded-For': ['ip1', 'ip2']}

        h = Headers(idict)
        self.assertDictEqual(dict(h),
                             {b'Content-Type': [b'text/html'],
                              b'X-Forwarded-For': [b'ip1', b'ip2']})
        self.assertSortedEqual(h.keys(),
                               [b'X-Forwarded-For', b'Content-Type'])
        self.assertSortedEqual(h.items(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.iteritems(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.values(), [b'ip2', b'text/html'])

    def test_update(self):
        h = Headers()
        h.update({'Content-Type': 'text/html',
                  'X-Forwarded-For': ['ip1', 'ip2']})
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])

    def test_copy(self):
        h1 = Headers({'header1': ['value1', 'value2']})
        h2 = copy.copy(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(h1.getlist('header1'), h2.getlist('header1'))
        assert h1.getlist('header1') is not h2.getlist('header1')
        assert isinstance(h2, Headers)

    def test_appendlist(self):
        h1 = Headers({'header1': 'value1'})
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

        h1 = Headers()
        h1.appendlist('header1', 'value1')
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

    def test_setlist(self):
        h1 = Headers({'header1': 'value1'})
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        h1.setlist('header1', [b'value2', b'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value2', b'value3'])

    def test_setlistdefault(self):
        h1 = Headers({'header1': 'value1'})
        h1.setlistdefault('header1', ['value2', 'value3'])
        h1.setlistdefault('header2', ['value2', 'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        self.assertEqual(h1.getlist('header2'), [b'value2', b'value3'])

    def test_none_value(self):
        h1 = Headers()
        h1['foo'] = 'bar'
        h1['foo'] = None
        h1.setdefault('foo', 'bar')
        self.assertEqual(h1.get('foo'), None)
        self.assertEqual(h1.getlist('foo'), [])

    def test_int_value(self):
        h1 = Headers({'hey': 5})
        h1['foo'] = 1
        h1.setdefault('bar', 2)
        h1.setlist('buz', [1, 'dos', 3])
        self.assertEqual(h1.getlist('foo'), [b'1'])
        self.assertEqual(h1.getlist('bar'), [b'2'])
        self.assertEqual(h1.getlist('buz'), [b'1', b'dos', b'3'])
        self.assertEqual(h1.getlist('hey'), [b'5'])

    def test_invalid_value(self):
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers, {'foo': object()})
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().__setitem__, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setdefault, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setlist, 'foo', [object()])

class M1(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process(self, response, request, spider):
        pass

class JsonLinesItemExporterTest(BaseItemExporterTest):

    _expected_nested = {'name': u'Jesus', 'age': {'name': 'Maria', 'age': {'name': 'Joseph', 'age': '22'}}}

    def _get_exporter(self, **kwargs):
        return JsonLinesItemExporter(self.output, **kwargs)

    def _check_output(self):
        exported = json.loads(to_unicode(self.output.getvalue().strip()))
        self.assertEqual(exported, dict(self.i))

    def test_nested_item(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = dict(name=u'Maria', age=i1)
        i3 = TestItem(name=u'Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, self._expected_nested)

    def test_extra_keywords(self):
        self.ie = self._get_exporter(sort_keys=True)
        self.test_export_item()
        self._check_output()
        self.assertRaises(TypeError, self._get_exporter, foo_unknown_keyword_bar=True)

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        item['time'] = str(item['time'])
        self.assertEqual(exported, item)


class FTPTestCase(BaseFTPTestCase):

    def test_invalid_credentials(self):
        from twisted.protocols.ftp import ConnectionLost

        meta = dict(self.req_meta)
        meta.update({"ftp_password": 'invalid'})
        request = Request(url="ftp://127.0.0.1:%s/file.txt" % self.portNum,
                          meta=meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            self.assertEqual(r.type, ConnectionLost)
        return self._add_test_callbacks(d, errback=_test)


class StartprojectTemplatesTest(ProjectTest):

    def setUp(self):
        super(StartprojectTemplatesTest, self).setUp()
        self.tmpl = join(self.temp_path, 'templates')
        self.tmpl_proj = join(self.tmpl, 'project')

    def test_startproject_template_override(self):
        copytree(join(scrapy.__path__[0], 'templates'), self.tmpl)
        with open(join(self.tmpl_proj, 'root_template'), 'w'):
            pass
        assert exists(join(self.tmpl_proj, 'root_template'))

        args = ['--set', 'TEMPLATES_DIR=%s' % self.tmpl]
        p = self.proc('startproject', self.project_name, *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("New Scrapy project %r, using template directory" % self.project_name, out)
        self.assertIn(self.tmpl_proj, out)
        assert exists(join(self.proj_path, 'root_template'))


class DictItemsSpider(TestSpider):
    item_cls = dict


def start_test_site(debug=False):
    root_dir = os.path.join(tests_datadir, "test_site")
    r = static.File(root_dir)
    r.putChild(b"redirect", util.Redirect(b"/redirected"))
    r.putChild(b"redirected", static.Data(b"Redirected here", "text/plain"))

    port = reactor.listenTCP(0, server.Site(r), interface="127.0.0.1")
    if debug:
        print("Test server running at http://localhost:%d/ - hit Ctrl-C to finish." \
            % port.getHost().port)
    return port


class JsonLinesItemExporterTest(BaseItemExporterTest):

    _expected_nested = {'name': u'Jesus', 'age': {'name': 'Maria', 'age': {'name': 'Joseph', 'age': '22'}}}

    def _get_exporter(self, **kwargs):
        return JsonLinesItemExporter(self.output, **kwargs)

    def _check_output(self):
        exported = json.loads(to_unicode(self.output.getvalue().strip()))
        self.assertEqual(exported, dict(self.i))

    def test_nested_item(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = dict(name=u'Maria', age=i1)
        i3 = TestItem(name=u'Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, self._expected_nested)

    def test_extra_keywords(self):
        self.ie = self._get_exporter(sort_keys=True)
        self.test_export_item()
        self._check_output()
        self.assertRaises(TypeError, self._get_exporter, foo_unknown_keyword_bar=True)

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        item['time'] = str(item['time'])
        self.assertEqual(exported, item)


class RedirectedMediaDownloadSpider(MediaDownloadSpider):
    name = 'redirectedmedia'

    def _process_url(self, url):
        return add_or_replace_parameter(
                    'http://localhost:8998/redirect-to',
                    'goto', url)


class MockedMediaPipeline(MediaPipeline):

    def __init__(self, *args, **kwargs):
        super(MockedMediaPipeline, self).__init__(*args, **kwargs)
        self._mockcalled = []

    def download(self, request, info):
        self._mockcalled.append('download')
        return super(MockedMediaPipeline, self).download(request, info)

    def media_to_download(self, request, info):
        self._mockcalled.append('media_to_download')
        if 'result' in request.meta:
            return request.meta.get('result')
        return super(MockedMediaPipeline, self).media_to_download(request, info)

    def get_media_requests(self, item, info):
        self._mockcalled.append('get_media_requests')
        return item.get('requests')

    def media_downloaded(self, response, request, info):
        self._mockcalled.append('media_downloaded')
        return super(MockedMediaPipeline, self).media_downloaded(response, request, info)

    def media_failed(self, failure, request, info):
        self._mockcalled.append('media_failed')
        return super(MockedMediaPipeline, self).media_failed(failure, request, info)

    def item_completed(self, results, item, info):
        self._mockcalled.append('item_completed')
        item = super(MockedMediaPipeline, self).item_completed(results, item, info)
        item['results'] = results
        return item


class LogformatterSubclassTest(LoggingContribTest):
    def setUp(self):
        self.formatter = LogFormatterSubclass()
        self.spider = Spider('default')

    def test_flags_in_request(self):
        pass


if __name__ == "__main__":
    unittest.main()

class MixinUnsafeUrl(object):
    scenarii = [
        # TLS to TLS: send referrer
        ('https://example.com/sekrit.html',     'http://not.example.com/',      b'https://example.com/sekrit.html'),
        ('https://example1.com/page.html',      'https://not.example1.com/',    b'https://example1.com/page.html'),
        ('https://example1.com/page.html',      'https://scrapy.org/',          b'https://example1.com/page.html'),
        ('https://example1.com:443/page.html',  'https://scrapy.org/',          b'https://example1.com/page.html'),
        ('https://example1.com:444/page.html',  'https://scrapy.org/',          b'https://example1.com:444/page.html'),
        ('ftps://example1.com/urls.zip',        'https://scrapy.org/',          b'ftps://example1.com/urls.zip'),

        # TLS to non-TLS: send referrer (yes, it's unsafe)
        ('https://example2.com/page.html',  'http://not.example2.com/', b'https://example2.com/page.html'),
        ('https://example2.com/page.html',  'http://scrapy.org/',       b'https://example2.com/page.html'),
        ('ftps://example2.com/urls.zip',    'http://scrapy.org/',       b'ftps://example2.com/urls.zip'),

        # non-TLS to TLS or non-TLS: send referrer (yes, it's unsafe)
        ('http://example3.com/page.html',       'https://not.example3.com/',    b'http://example3.com/page.html'),
        ('http://example3.com/page.html',       'https://scrapy.org/',          b'http://example3.com/page.html'),
        ('http://example3.com:8080/page.html',  'https://scrapy.org/',          b'http://example3.com:8080/page.html'),
        ('http://example3.com:80/page.html',    'http://not.example3.com/',     b'http://example3.com/page.html'),
        ('http://example3.com/page.html',       'http://scrapy.org/',           b'http://example3.com/page.html'),
        ('http://example3.com:443/page.html',   'http://scrapy.org/',           b'http://example3.com:443/page.html'),
        ('ftp://example3.com/urls.zip',         'http://scrapy.org/',           b'ftp://example3.com/urls.zip'),
        ('ftp://example3.com/urls.zip',         'https://scrapy.org/',          b'ftp://example3.com/urls.zip'),

        # test for user/password stripping
        ('http://user:password@example4.com/page.html',     'https://not.example4.com/',    b'http://example4.com/page.html'),
        ('https://user:password@example4.com/page.html',    'http://scrapy.org/',           b'https://example4.com/page.html'),
    ]


class CrawlerProcessTest(BaseCrawlerTest):
    def test_crawler_process_accepts_dict(self):
        runner = CrawlerProcess({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_process_accepts_None(self):
        runner = CrawlerProcess()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

class AnonymousFTPTestCase(BaseFTPTestCase):

    username = "anonymous"
    req_meta = {}

    def setUp(self):
        from twisted.protocols.ftp import FTPRealm, FTPFactory
        from scrapy.core.downloader.handlers.ftp import FTPDownloadHandler

        # setup dir and test file
        self.directory = self.mktemp()
        os.mkdir(self.directory)

        fp = FilePath(self.directory)
        fp.child('file.txt').setContent(b"I have the power!")
        fp.child('file with spaces.txt').setContent(b"Moooooooooo power!")

        # setup server for anonymous access
        realm = FTPRealm(anonymousRoot=self.directory)
        p = portal.Portal(realm)
        p.registerChecker(checkers.AllowAnonymousAccess(),
                          credentials.IAnonymous)

        self.factory = FTPFactory(portal=p,
                                  userAnonymous=self.username)
        self.port = reactor.listenTCP(0, self.factory, interface="127.0.0.1")
        self.portNum = self.port.getHost().port
        self.download_handler = FTPDownloadHandler(Settings())
        self.addCleanup(self.port.stopListening)

    def tearDown(self):
        shutil.rmtree(self.directory)


class TestS3FilesStore(unittest.TestCase):
    @defer.inlineCallbacks
    def test_persist(self):
        assert_aws_environ()
        uri = os.environ.get('S3_TEST_FILE_URI')
        if not uri:
            raise unittest.SkipTest("No S3 URI available for testing")
        data = b"TestS3FilesStore: \xe2\x98\x83"
        buf = BytesIO(data)
        meta = {'foo': 'bar'}
        path = ''
        store = S3FilesStore(uri)
        yield store.persist_file(
            path, buf, info=None, meta=meta,
            headers={'Content-Type': 'image/png'})
        s = yield store.stat_file(path, info=None)
        self.assertIn('last_modified', s)
        self.assertIn('checksum', s)
        self.assertEqual(s['checksum'], '3187896a9657a28163abb31667df64c8')
        u = urlparse(uri)
        content, key = get_s3_content_and_delete(
            u.hostname, u.path[1:], with_key=True)
        self.assertEqual(content, data)
        if is_botocore():
            self.assertEqual(key['Metadata'], {'foo': 'bar'})
            self.assertEqual(
                key['CacheControl'], S3FilesStore.HEADERS['Cache-Control'])
            self.assertEqual(key['ContentType'], 'image/png')
        else:
            self.assertEqual(key.metadata, {'foo': 'bar'})
            self.assertEqual(
                key.cache_control, S3FilesStore.HEADERS['Cache-Control'])
            self.assertEqual(key.content_type, 'image/png')


class FilesystemStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

class CustomItem(Item):

    name = Field()

    def __str__(self):
        return "name: %s" % self['name']


class SendCatchLogTest2(unittest.TestCase):

    def test_error_logged_if_deferred_not_supported(self):
        test_signal = object()
        test_handler = lambda: defer.Deferred()
        dispatcher.connect(test_handler, test_signal)
        with LogCapture() as l:
            send_catch_log(test_signal)
        self.assertEqual(len(l.records), 1)
        self.assertIn("Cannot return deferreds from signal handler", str(l))
        dispatcher.disconnect(test_handler, test_signal)

class WrappedRequestTest(TestCase):

    def setUp(self):
        self.request = Request("http://www.example.com/page.html",
                               headers={"Content-Type": "text/html"})
        self.wrapped = WrappedRequest(self.request)

    def test_get_full_url(self):
        self.assertEqual(self.wrapped.get_full_url(), self.request.url)
        self.assertEqual(self.wrapped.full_url, self.request.url)

    def test_get_host(self):
        self.assertEqual(self.wrapped.get_host(), urlparse(self.request.url).netloc)
        self.assertEqual(self.wrapped.host, urlparse(self.request.url).netloc)

    def test_get_type(self):
        self.assertEqual(self.wrapped.get_type(), urlparse(self.request.url).scheme)
        self.assertEqual(self.wrapped.type, urlparse(self.request.url).scheme)

    def test_is_unverifiable(self):
        self.assertFalse(self.wrapped.is_unverifiable())
        self.assertFalse(self.wrapped.unverifiable)

    def test_is_unverifiable2(self):
        self.request.meta['is_unverifiable'] = True
        self.assertTrue(self.wrapped.is_unverifiable())
        self.assertTrue(self.wrapped.unverifiable)

    def test_get_origin_req_host(self):
        self.assertEqual(self.wrapped.get_origin_req_host(), 'www.example.com')
        self.assertEqual(self.wrapped.origin_req_host, 'www.example.com')

    def test_has_header(self):
        self.assertTrue(self.wrapped.has_header('content-type'))
        self.assertFalse(self.wrapped.has_header('xxxxx'))

    def test_get_header(self):
        self.assertEqual(self.wrapped.get_header('content-type'), 'text/html')
        self.assertEqual(self.wrapped.get_header('xxxxx', 'def'), 'def')

    def test_header_items(self):
        self.assertEqual(self.wrapped.header_items(),
                         [('Content-Type', ['text/html'])])

    def test_add_unredirected_header(self):
        self.wrapped.add_unredirected_header('hello', 'world')
        self.assertEqual(self.request.headers['hello'], b'world')

