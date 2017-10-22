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


class ItemTest(unittest.TestCase):

    def assertSortedEqual(self, first, second, msg=None):
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_simple(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i['name'] = u'name'
        self.assertEqual(i['name'], u'name')

    def test_init(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(KeyError, i.__getitem__, 'name')

        i2 = TestItem(name=u'john doe')
        self.assertEqual(i2['name'], u'john doe')

        i3 = TestItem({'name': u'john doe'})
        self.assertEqual(i3['name'], u'john doe')

        i4 = TestItem(i3)
        self.assertEqual(i4['name'], u'john doe')

        self.assertRaises(KeyError, TestItem, {'name': u'john doe',
                                               'other': u'foo'})

    def test_invalid_field(self):
        class TestItem(Item):
            pass

        i = TestItem()
        self.assertRaises(KeyError, i.__setitem__, 'field', 'text')
        self.assertRaises(KeyError, i.__getitem__, 'field')

    def test_repr(self):
        class TestItem(Item):
            name = Field()
            number = Field()

        i = TestItem()
        i['name'] = u'John Doe'
        i['number'] = 123
        itemrepr = repr(i)

        if six.PY2:
            self.assertEqual(itemrepr,
                             "{'name': u'John Doe', 'number': 123}")
        else:
            self.assertEqual(itemrepr,
                             "{'name': 'John Doe', 'number': 123}")

        i2 = eval(itemrepr)
        self.assertEqual(i2['name'], 'John Doe')
        self.assertEqual(i2['number'], 123)

    def test_private_attr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i._private = 'test'
        self.assertEqual(i._private, 'test')

    def test_raise_getattr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(AttributeError, getattr, i, 'name')

    def test_raise_setattr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(AttributeError, setattr, i, 'name', 'john')

    def test_custom_methods(self):
        class TestItem(Item):
            name = Field()

            def get_name(self):
                return self['name']

            def change_name(self, name):
                self['name'] = name

        i = TestItem()
        self.assertRaises(KeyError, i.get_name)
        i['name'] = u'lala'
        self.assertEqual(i.get_name(), u'lala')
        i.change_name(u'other')
        self.assertEqual(i.get_name(), 'other')

    def test_metaclass(self):
        class TestItem(Item):
            name = Field()
            keys = Field()
            values = Field()

        i = TestItem()
        i['name'] = u'John'
        self.assertEqual(list(i.keys()), ['name'])
        self.assertEqual(list(i.values()), ['John'])

        i['keys'] = u'Keys'
        i['values'] = u'Values'
        self.assertSortedEqual(list(i.keys()), ['keys', 'values', 'name'])
        self.assertSortedEqual(list(i.values()), [u'Keys', u'Values', u'John'])

    def test_metaclass_with_fields_attribute(self):
        class TestItem(Item):
            fields = {'new': Field(default='X')}

        item = TestItem(new=u'New')
        self.assertSortedEqual(list(item.keys()), ['new'])
        self.assertSortedEqual(list(item.values()), [u'New'])

    def test_metaclass_inheritance(self):
        class BaseItem(Item):
            name = Field()
            keys = Field()
            values = Field()

        class TestItem(BaseItem):
            keys = Field()

        i = TestItem()
        i['keys'] = 3
        self.assertEqual(list(i.keys()), ['keys'])
        self.assertEqual(list(i.values()), [3])

    def test_metaclass_multiple_inheritance_simple(self):
        class A(Item):
            fields = {'load': Field(default='A')}
            save = Field(default='A')

        class B(A): pass

        class C(Item):
            fields = {'load': Field(default='C')}
            save = Field(default='C')

        class D(B, C): pass

        item = D(save='X', load='Y')
        self.assertEqual(item['save'], 'X')
        self.assertEqual(item['load'], 'Y')
        self.assertEqual(D.fields, {'load': {'default': 'A'},
            'save': {'default': 'A'}})

        # D class inverted
        class E(C, B): pass

        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E(load='X')['load'], 'X')
        self.assertEqual(E.fields, {'load': {'default': 'C'},
            'save': {'default': 'C'}})

    def test_metaclass_multiple_inheritance_diamond(self):
        class A(Item):
            fields = {'update': Field(default='A')}
            save = Field(default='A')
            load = Field(default='A')

        class B(A): pass

        class C(A):
            fields = {'update': Field(default='C')}
            save = Field(default='C')

        class D(B, C):
            fields = {'update': Field(default='D')}
            load = Field(default='D')

        self.assertEqual(D(save='X')['save'], 'X')
        self.assertEqual(D(load='X')['load'], 'X')
        self.assertEqual(D.fields, {'save': {'default': 'C'},
            'load': {'default': 'D'}, 'update': {'default': 'D'}})

        # D class inverted
        class E(C, B):
            load = Field(default='E')

        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E(load='X')['load'], 'X')
        self.assertEqual(E.fields, {'save': {'default': 'C'},
            'load': {'default': 'E'}, 'update': {'default': 'C'}})

    def test_metaclass_multiple_inheritance_without_metaclass(self):
        class A(Item):
            fields = {'load': Field(default='A')}
            save = Field(default='A')

        class B(A): pass

        class C(object):
            fields = {'load': Field(default='C')}
            not_allowed = Field(default='not_allowed')
            save = Field(default='C')

        class D(B, C): pass

        self.assertRaises(KeyError, D, not_allowed='value')
        self.assertEqual(D(save='X')['save'], 'X')
        self.assertEqual(D.fields, {'save': {'default': 'A'},
            'load': {'default': 'A'}})

        # D class inverted
        class E(C, B): pass

        self.assertRaises(KeyError, E, not_allowed='value')
        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E.fields, {'save': {'default': 'A'},
            'load': {'default': 'A'}})

    def test_to_dict(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i['name'] = u'John'
        self.assertEqual(dict(i), {'name': u'John'})

    def test_copy(self):
        class TestItem(Item):
            name = Field()
        item = TestItem({'name':'lower'})
        copied_item = item.copy()
        self.assertNotEqual(id(item), id(copied_item))
        copied_item['name'] = copied_item['name'].upper()
        self.assertNotEqual(item['name'], copied_item['name'])


class ImagesPipelineTestCaseFields(unittest.TestCase):

    def test_item_fields_default(self):
        class TestItem(Item):
            name = Field()
            image_urls = Field()
            images = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/images/1.jpg'
            item = cls({'name': 'item1', 'image_urls': [url]})
            pipeline = ImagesPipeline.from_settings(Settings({'IMAGES_STORE': 's3://example/images/'}))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['images'], [results[0][1]])

    def test_item_fields_override_settings(self):
        class TestItem(Item):
            name = Field()
            image = Field()
            stored_image = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/images/1.jpg'
            item = cls({'name': 'item1', 'image': [url]})
            pipeline = ImagesPipeline.from_settings(Settings({
                'IMAGES_STORE': 's3://example/images/',
                'IMAGES_URLS_FIELD': 'image',
                'IMAGES_RESULT_FIELD': 'stored_image'
            }))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['stored_image'], [results[0][1]])


class NameItemLoader(ItemLoader):
    default_item_class = TestItem


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


def _test_data(formats):
    uncompressed_body = get_testdata('compressed', 'feed-sample1.xml')
    test_responses = {}
    for format in formats:
        body = get_testdata('compressed', 'feed-sample1.' + format)
        test_responses[format] = Response('http://foo.com/bar', body=body)
    return uncompressed_body, test_responses


class Http11MockServerTestCase(unittest.TestCase):
    """HTTP 1.1 test case with MockServer"""

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_download_with_content_length(self):
        crawler = get_crawler(SingleRequestSpider)
        # http://localhost:8998/partial set Content-Length to 1024, use download_maxsize= 1000 to avoid
        # download it
        yield crawler.crawl(seed=Request(url='http://localhost:8998/partial', meta={'download_maxsize': 1000}))
        failure = crawler.spider.meta['failure']
        self.assertIsInstance(failure.value, defer.CancelledError)

    @defer.inlineCallbacks
    def test_download(self):
        crawler = get_crawler(SingleRequestSpider)
        yield crawler.crawl(seed=Request(url='http://localhost:8998'))
        failure = crawler.spider.meta.get('failure')
        self.assertTrue(failure == None)
        reason = crawler.spider.meta['close_reason']
        self.assertTrue(reason, 'finished')

    @defer.inlineCallbacks
    def test_download_gzip_response(self):
        crawler = get_crawler(SingleRequestSpider)
        body = b'1' * 100  # PayloadResource requires body length to be 100
        request = Request('http://localhost:8998/payload', method='POST',
                          body=body, meta={'download_maxsize': 50})
        yield crawler.crawl(seed=request)
        failure = crawler.spider.meta['failure']
        # download_maxsize < 100, hence the CancelledError
        self.assertIsInstance(failure.value, defer.CancelledError)

        if six.PY2:
            request.headers.setdefault(b'Accept-Encoding', b'gzip,deflate')
            request = request.replace(url='http://localhost:8998/xpayload')
            yield crawler.crawl(seed=request)
            # download_maxsize = 50 is enough for the gzipped response
            failure = crawler.spider.meta.get('failure')
            self.assertTrue(failure == None)
            reason = crawler.spider.meta['close_reason']
            self.assertTrue(reason, 'finished')
        else:
            # See issue https://twistedmatrix.com/trac/ticket/8175
            raise unittest.SkipTest("xpayload only enabled for PY2")


class Base:
    class LinkExtractorTestCase(unittest.TestCase):
        extractor_cls = None
        escapes_whitespace = False

        def setUp(self):
            body = get_testdata('link_extractor', 'sgml_linkextractor.html')
            self.response = HtmlResponse(url='http://example.com/index', body=body)

        def test_urls_type(self):
            ''' Test that the resulting urls are str objects '''
            lx = self.extractor_cls()
            self.assertTrue(all(isinstance(link.url, str)
                                for link in lx.extract_links(self.response)))

        def test_extract_all_links(self):
            lx = self.extractor_cls()
            if self.escapes_whitespace:
                page4_url = 'http://example.com/page%204.html'
            else:
                page4_url = 'http://example.com/page 4.html'

            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

        def test_extract_filter_allow(self):
            lx = self.extractor_cls(allow=('sample', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_with_duplicates(self):
            lx = self.extractor_cls(allow=('sample', ), unique=False)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 repetition'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_with_duplicates_canonicalize(self):
            lx = self.extractor_cls(allow=('sample', ), unique=False,
                                    canonicalize=True)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 repetition'),
                Link(url='http://example.com/sample3.html', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_no_duplicates_canonicalize(self):
            lx = self.extractor_cls(allow=('sample',), unique=True,
                                    canonicalize=True)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
            ])

        def test_extract_filter_allow_and_deny(self):
            lx = self.extractor_cls(allow=('sample', ), deny=('3', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

        def test_extract_filter_allowed_domains(self):
            lx = self.extractor_cls(allow_domains=('google.com', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

        def test_extraction_using_single_values(self):
            '''Test the extractor's behaviour among different situations'''

            lx = self.extractor_cls(allow='sample')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo',
                     text='sample 3 repetition with fragment')
            ])

            lx = self.extractor_cls(allow='sample', deny='3')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(allow_domains='google.com')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

            lx = self.extractor_cls(deny_domains='example.com')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

        def test_nofollow(self):
            '''Test the extractor's behaviour for links with rel="nofollow"'''

            html = b"""<html><head><title>Page title<title>
            <body>
            <div class='links'>
            <p><a href="/about.html">About us</a></p>
            </div>
            <div>
            <p><a href="/follow.html">Follow this link</a></p>
            </div>
            <div>
            <p><a href="/nofollow.html" rel="nofollow">Dont follow this one</a></p>
            </div>
            <div>
            <p><a href="/nofollow2.html" rel="blah">Choose to follow or not</a></p>
            </div>
            <div>
            <p><a href="http://google.com/something" rel="external nofollow">External link not to follow</a></p>
            </div>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/about.html', text=u'About us'),
                Link(url='http://example.org/follow.html', text=u'Follow this link'),
                Link(url='http://example.org/nofollow.html', text=u'Dont follow this one', nofollow=True),
                Link(url='http://example.org/nofollow2.html', text=u'Choose to follow or not'),
                Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True),
            ])

        def test_matches(self):
            url1 = 'http://lotsofstuff.com/stuff1/index'
            url2 = 'http://evenmorestuff.com/uglystuff/index'

            lx = self.extractor_cls(allow=(r'stuff1', ))
            self.assertEqual(lx.matches(url1), True)
            self.assertEqual(lx.matches(url2), False)

            lx = self.extractor_cls(deny=(r'uglystuff', ))
            self.assertEqual(lx.matches(url1), True)
            self.assertEqual(lx.matches(url2), False)

            lx = self.extractor_cls(allow_domains=('evenmorestuff.com', ))
            self.assertEqual(lx.matches(url1), False)
            self.assertEqual(lx.matches(url2), True)

            lx = self.extractor_cls(deny_domains=('lotsofstuff.com', ))
            self.assertEqual(lx.matches(url1), False)
            self.assertEqual(lx.matches(url2), True)

            lx = self.extractor_cls(allow=('blah1',), deny=('blah2',),
                                   allow_domains=('blah1.com',),
                                   deny_domains=('blah2.com',))
            self.assertEqual(lx.matches('http://blah1.com/blah1'), True)
            self.assertEqual(lx.matches('http://blah1.com/blah2'), False)
            self.assertEqual(lx.matches('http://blah2.com/blah1'), False)
            self.assertEqual(lx.matches('http://blah2.com/blah2'), False)

        def test_restrict_xpaths(self):
            lx = self.extractor_cls(restrict_xpaths=('//div[@id="subwrapper"]', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

        def test_restrict_xpaths_encoding(self):
            """Test restrict_xpaths with encodings"""
            html = b"""<html><head><title>Page title<title>
            <body><p><a href="item/12.html">Item 12</a></p>
            <div class='links'>
            <p><a href="/about.html">About us\xa3</a></p>
            </div>
            <div>
            <p><a href="/nofollow.html">This shouldn't be followed</a></p>
            </div>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='windows-1252')

            lx = self.extractor_cls(restrict_xpaths="//div[@class='links']")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/about.html', text=u'About us\xa3')])

        def test_restrict_xpaths_with_html_entities(self):
            html = b'<html><body><p><a href="/&hearts;/you?c=&euro;">text</a></p></body></html>'
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='iso8859-15')
            links = self.extractor_cls(restrict_xpaths='//p').extract_links(response)
            self.assertEqual(links,
                             [Link(url='http://example.org/%E2%99%A5/you?c=%E2%82%AC', text=u'text')])

        def test_restrict_xpaths_concat_in_handle_data(self):
            """html entities cause SGMLParser to call handle_data hook twice"""
            body = b"""<html><body><div><a href="/foo">&gt;\xbe\xa9&lt;\xb6\xab</a></body></html>"""
            response = HtmlResponse("http://example.org", body=body, encoding='gb18030')
            lx = self.extractor_cls(restrict_xpaths="//div")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/foo', text=u'>\u4eac<\u4e1c',
                                   fragment='', nofollow=False)])

        def test_restrict_css(self):
            lx = self.extractor_cls(restrict_css=('#subwrapper a',))
            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2')
            ])

        def test_restrict_css_and_restrict_xpaths_together(self):
            lx = self.extractor_cls(restrict_xpaths=('//div[@id="subwrapper"]', ),
                                    restrict_css=('#subwrapper + a', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
            ])

        def test_area_tag_with_unicode_present(self):
            body = b"""<html><body>\xbe\xa9<map><area href="http://example.org/foo" /></map></body></html>"""
            response = HtmlResponse("http://example.org", body=body, encoding='utf-8')
            lx = self.extractor_cls()
            lx.extract_links(response)
            lx.extract_links(response)
            lx.extract_links(response)
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/foo', text=u'',
                                   fragment='', nofollow=False)])

        def test_encoded_url(self):
            body = b"""<html><body><div><a href="?page=2">BinB</a></body></html>"""
            response = HtmlResponse("http://known.fm/AC%2FDC/", body=body, encoding='utf8')
            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://known.fm/AC%2FDC/?page=2', text=u'BinB', fragment='', nofollow=False),
            ])

        def test_encoded_url_in_restricted_xpath(self):
            body = b"""<html><body><div><a href="?page=2">BinB</a></body></html>"""
            response = HtmlResponse("http://known.fm/AC%2FDC/", body=body, encoding='utf8')
            lx = self.extractor_cls(restrict_xpaths="//div")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://known.fm/AC%2FDC/?page=2', text=u'BinB', fragment='', nofollow=False),
            ])

        def test_ignored_extensions(self):
            # jpg is ignored by default
            html = b"""<a href="page.html">asd</a> and <a href="photo.jpg">"""
            response = HtmlResponse("http://example.org/", body=html)
            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/page.html', text=u'asd'),
            ])

            # override denied extensions
            lx = self.extractor_cls(deny_extensions=['html'])
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/photo.jpg'),
            ])

        def test_process_value(self):
            """Test restrict_xpaths with encodings"""
            html = b"""
            <a href="javascript:goToPage('../other/page.html','photo','width=600,height=540,scrollbars'); return false">Link text</a>
            <a href="/about.html">About us</a>
            """
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='windows-1252')

            def process_value(value):
                m = re.search("javascript:goToPage\('(.*?)'", value)
                if m:
                    return m.group(1)

            lx = self.extractor_cls(process_value=process_value)
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/other/page.html', text='Link text')])

        def test_base_url_with_restrict_xpaths(self):
            html = b"""<html><head><title>Page title<title><base href="http://otherdomain.com/base/" />
            <body><p><a href="item/12.html">Item 12</a></p>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html)
            lx = self.extractor_cls(restrict_xpaths="//p")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://otherdomain.com/base/item/12.html', text='Item 12')])

        def test_attrs(self):
            lx = self.extractor_cls(attrs="href")
            if self.escapes_whitespace:
                page4_url = 'http://example.com/page%204.html'
            else:
                page4_url = 'http://example.com/page 4.html'

            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

            lx = self.extractor_cls(attrs=("href","src"), tags=("a","area","img"), deny_extensions=())
            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample2.jpg', text=u''),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

            lx = self.extractor_cls(attrs=None)
            self.assertEqual(lx.extract_links(self.response), [])

        def test_tags(self):
            html = b"""<html><area href="sample1.html"></area><a href="sample2.html">sample 2</a><img src="sample2.jpg"/></html>"""
            response = HtmlResponse("http://example.com/index.html", body=html)

            lx = self.extractor_cls(tags=None)
            self.assertEqual(lx.extract_links(response), [])

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(tags="area")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample1.html', text=u''),
            ])

            lx = self.extractor_cls(tags="a")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(tags=("a","img"), attrs=("href", "src"), deny_extensions=())
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample2.jpg', text=u''),
            ])

        def test_tags_attrs(self):
            html = b"""
            <html><body>
            <div id="item1" data-url="get?id=1"><a href="#">Item 1</a></div>
            <div id="item2" data-url="get?id=2"><a href="#">Item 2</a></div>
            </body></html>
            """
            response = HtmlResponse("http://example.com/index.html", body=html)

            lx = self.extractor_cls(tags='div', attrs='data-url')
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/get?id=1', text=u'Item 1', fragment='', nofollow=False),
                Link(url='http://example.com/get?id=2', text=u'Item 2', fragment='', nofollow=False)
            ])

            lx = self.extractor_cls(tags=('div',), attrs=('data-url',))
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/get?id=1', text=u'Item 1', fragment='', nofollow=False),
                Link(url='http://example.com/get?id=2', text=u'Item 2', fragment='', nofollow=False)
            ])

        def test_xhtml(self):
            xhtml = b"""
    <?xml version="1.0"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
        <title>XHTML document title</title>
    </head>
    <body>
        <div class='links'>
        <p><a href="/about.html">About us</a></p>
        </div>
        <div>
        <p><a href="/follow.html">Follow this link</a></p>
        </div>
        <div>
        <p><a href="/nofollow.html" rel="nofollow">Dont follow this one</a></p>
        </div>
        <div>
        <p><a href="/nofollow2.html" rel="blah">Choose to follow or not</a></p>
        </div>
        <div>
        <p><a href="http://google.com/something" rel="external nofollow">External link not to follow</a></p>
        </div>
    </body>
    </html>
            """

            response = HtmlResponse("http://example.com/index.xhtml", body=xhtml)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.com/about.html', text=u'About us', fragment='', nofollow=False),
                              Link(url='http://example.com/follow.html', text=u'Follow this link', fragment='', nofollow=False),
                              Link(url='http://example.com/nofollow.html', text=u'Dont follow this one', fragment='', nofollow=True),
                              Link(url='http://example.com/nofollow2.html', text=u'Choose to follow or not', fragment='', nofollow=False),
                              Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True)]
                            )

            response = XmlResponse("http://example.com/index.xhtml", body=xhtml)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.com/about.html', text=u'About us', fragment='', nofollow=False),
                              Link(url='http://example.com/follow.html', text=u'Follow this link', fragment='', nofollow=False),
                              Link(url='http://example.com/nofollow.html', text=u'Dont follow this one', fragment='', nofollow=True),
                              Link(url='http://example.com/nofollow2.html', text=u'Choose to follow or not', fragment='', nofollow=False),
                              Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True)]
                            )

        def test_link_wrong_href(self):
            html = b"""
            <a href="http://example.org/item1.html">Item 1</a>
            <a href="http://[example.org/item2.html">Item 2</a>
            <a href="http://example.org/item3.html">Item 3</a>
            """
            response = HtmlResponse("http://example.org/index.html", body=html)
            lx = self.extractor_cls()
            self.assertEqual([link for link in lx.extract_links(response)], [
                Link(url='http://example.org/item1.html', text=u'Item 1', nofollow=False),
                Link(url='http://example.org/item3.html', text=u'Item 3', nofollow=False),
            ])


class LogFormatterSubclass(LogFormatter):
    def crawled(self, request, response, spider):
        kwargs = super(LogFormatterSubclass, self).crawled(
        request, response, spider)
        CRAWLEDMSG = (
            u"Crawled (%(status)s) %(request)s (referer: "
            u"%(referer)s)%(flags)s"
        )
        return {
            'level': kwargs['level'],
            'msg': CRAWLEDMSG,
            'args': kwargs['args']
        }


class ImagesPipelineTestCaseCustomSettings(unittest.TestCase):
    img_cls_attribute_names = [
        # Pipeline attribute names with corresponding setting names.
        ("EXPIRES", "IMAGES_EXPIRES"),
        ("MIN_WIDTH", "IMAGES_MIN_WIDTH"),
        ("MIN_HEIGHT", "IMAGES_MIN_HEIGHT"),
        ("IMAGES_URLS_FIELD", "IMAGES_URLS_FIELD"),
        ("IMAGES_RESULT_FIELD", "IMAGES_RESULT_FIELD"),
        ("THUMBS", "IMAGES_THUMBS")
    ]

    # This should match what is defined in ImagesPipeline.
    default_pipeline_settings = dict(
        MIN_WIDTH=0,
        MIN_HEIGHT=0,
        EXPIRES=90,
        THUMBS={},
        IMAGES_URLS_FIELD='image_urls',
        IMAGES_RESULT_FIELD='images'
    )


    def setUp(self):
        self.tempdir = mkdtemp()

    def tearDown(self):
        rmtree(self.tempdir)

    def _generate_fake_settings(self, prefix=None):
        """
        :param prefix: string for setting keys
        :return: dictionary of image pipeline settings
        """

        def random_string():
            return "".join([chr(random.randint(97, 123)) for _ in range(10)])

        settings = {
            "IMAGES_EXPIRES": random.randint(100, 1000),
            "IMAGES_STORE": self.tempdir,
            "IMAGES_RESULT_FIELD": random_string(),
            "IMAGES_URLS_FIELD": random_string(),
            "IMAGES_MIN_WIDTH": random.randint(1, 1000),
            "IMAGES_MIN_HEIGHT": random.randint(1, 1000),
            "IMAGES_THUMBS": {
                'small': (random.randint(1, 1000), random.randint(1, 1000)),
                'big': (random.randint(1, 1000), random.randint(1, 1000))
            }
        }
        if not prefix:
            return settings

        return {prefix.upper() + "_" + k if k != "IMAGES_STORE" else k: v for k, v in settings.items()}

    def _generate_fake_pipeline_subclass(self):
        """
        :return: ImagePipeline class will all uppercase attributes set.
        """
        class UserDefinedImagePipeline(ImagesPipeline):
            # Values should be in different range than fake_settings.
            MIN_WIDTH = random.randint(1000, 2000)
            MIN_HEIGHT = random.randint(1000, 2000)
            THUMBS = {
                'small': (random.randint(1000, 2000), random.randint(1000, 2000)),
                'big': (random.randint(1000, 2000), random.randint(1000, 2000))
            }
            EXPIRES = random.randint(1000, 2000)
            IMAGES_URLS_FIELD = "field_one"
            IMAGES_RESULT_FIELD = "field_two"

        return UserDefinedImagePipeline

    def test_different_settings_for_different_instances(self):
        """
        If there are two instances of ImagesPipeline class with different settings, they should
        have different settings.
        """
        custom_settings = self._generate_fake_settings()
        default_settings = Settings()
        default_sts_pipe = ImagesPipeline(self.tempdir, settings=default_settings)
        user_sts_pipe = ImagesPipeline.from_settings(Settings(custom_settings))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            expected_default_value = self.default_pipeline_settings.get(pipe_attr)
            custom_value = custom_settings.get(settings_attr)
            self.assertNotEqual(expected_default_value, custom_value)
            self.assertEqual(getattr(default_sts_pipe, pipe_attr.lower()), expected_default_value)
            self.assertEqual(getattr(user_sts_pipe, pipe_attr.lower()), custom_value)

    def test_subclass_attrs_preserved_default_settings(self):
        """
        If image settings are not defined at all subclass of ImagePipeline takes values
        from class attributes.
        """
        pipeline_cls = self._generate_fake_pipeline_subclass()
        pipeline = pipeline_cls.from_settings(Settings({"IMAGES_STORE": self.tempdir}))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            # Instance attribute (lowercase) must be equal to class attribute (uppercase).
            attr_value = getattr(pipeline, pipe_attr.lower())
            self.assertNotEqual(attr_value, self.default_pipeline_settings[pipe_attr])
            self.assertEqual(attr_value, getattr(pipeline, pipe_attr))

    def test_subclass_attrs_preserved_custom_settings(self):
        """
        If image settings are defined but they are not defined for subclass default
        values taken from settings should be preserved.
        """
        pipeline_cls = self._generate_fake_pipeline_subclass()
        settings = self._generate_fake_settings()
        pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            # Instance attribute (lowercase) must be equal to
            # value defined in settings.
            value = getattr(pipeline, pipe_attr.lower())
            self.assertNotEqual(value, self.default_pipeline_settings[pipe_attr])
            setings_value = settings.get(settings_attr)
            self.assertEqual(value, setings_value)

    def test_no_custom_settings_for_subclasses(self):
        """
        If there are no settings for subclass and no subclass attributes, pipeline should use
        attributes of base class.
        """
        class UserDefinedImagePipeline(ImagesPipeline):
            pass

        user_pipeline = UserDefinedImagePipeline.from_settings(Settings({"IMAGES_STORE": self.tempdir}))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = self.default_pipeline_settings.get(pipe_attr.upper())
            self.assertEqual(getattr(user_pipeline, pipe_attr.lower()), custom_value)

    def test_custom_settings_for_subclasses(self):
        """
        If there are custom settings for subclass and NO class attributes, pipeline should use custom
        settings.
        """
        class UserDefinedImagePipeline(ImagesPipeline):
            pass

        prefix = UserDefinedImagePipeline.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = UserDefinedImagePipeline.from_settings(Settings(settings))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_pipeline_settings[pipe_attr])
            self.assertEqual(getattr(user_pipeline, pipe_attr.lower()), custom_value)

    def test_custom_settings_and_class_attrs_for_subclasses(self):
        """
        If there are custom settings for subclass AND class attributes
        setting keys are preferred and override attributes.
        """
        pipeline_cls = self._generate_fake_pipeline_subclass()
        prefix = pipeline_cls.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_pipeline_settings[pipe_attr])
            self.assertEqual(getattr(user_pipeline, pipe_attr.lower()), custom_value)

    def test_cls_attrs_with_DEFAULT_prefix(self):
        class UserDefinedImagePipeline(ImagesPipeline):
            DEFAULT_IMAGES_URLS_FIELD = "something"
            DEFAULT_IMAGES_RESULT_FIELD = "something_else"
        pipeline = UserDefinedImagePipeline.from_settings(Settings({"IMAGES_STORE": self.tempdir}))
        self.assertEqual(pipeline.images_result_field, "something_else")
        self.assertEqual(pipeline.images_urls_field, "something")

    def test_user_defined_subclass_default_key_names(self):
        """Test situation when user defines subclass of ImagePipeline,
        but uses attribute names for default pipeline (without prefixing
        them with pipeline class name).
        """
        settings = self._generate_fake_settings()

        class UserPipe(ImagesPipeline):
            pass

        pipeline_cls = UserPipe.from_settings(Settings(settings))

        for pipe_attr, settings_attr in self.img_cls_attribute_names:
            expected_value = settings.get(settings_attr)
            self.assertEqual(getattr(pipeline_cls, pipe_attr.lower()),
                             expected_value)

def _create_image(format, *a, **kw):
    buf = TemporaryFile()
    Image.new(*a, **kw).save(buf, format)
    buf.seek(0)
    return Image.open(buf)


if __name__ == "__main__":
    unittest.main()

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

class ItemMetaTest(unittest.TestCase):

    def test_new_method_propagates_classcell(self):
        new_mock = mock.Mock(side_effect=ABCMeta.__new__)
        base = ItemMeta.__bases__[0]

        with mock.patch.object(base, '__new__', new_mock):

            class MyItem(Item):
                if not PY36_PLUS:
                    # This attribute is an internal attribute in Python 3.6+
                    # and must be propagated properly. See
                    # https://docs.python.org/3.6/reference/datamodel.html#creating-the-class-object
                    # In <3.6, we add a dummy attribute just to ensure the
                    # __new__ method propagates it correctly.
                    __classcell__ = object()

                def f(self):
                    # For rationale of this see:
                    # https://github.com/python/cpython/blob/ee1a81b77444c6715cbe610e951c655b6adab88b/Lib/test/test_super.py#L222
                    return __class__  # noqa  https://github.com/scrapy/scrapy/issues/2836

            MyItem()

        (first_call, second_call) = new_mock.call_args_list[-2:]

        mcs, class_name, bases, attrs = first_call[0]
        assert '__classcell__' not in attrs
        mcs, class_name, bases, attrs = second_call[0]
        assert '__classcell__' in attrs


class DeprecatedFilesPipeline(FilesPipeline):
    def file_key(self, url):
        media_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        media_ext = os.path.splitext(url)[1]
        return 'empty/%s%s' % (media_guid, media_ext)


class ChunkedResource(resource.Resource):

    def render(self, request):
        def response():
            request.write(b"chunked ")
            request.write(b"content\n")
            request.finish()
        reactor.callLater(0, response)
        return server.NOT_DONE_YET


class Https11WrongHostnameTestCase(Http11TestCase):
    scheme = 'https'

    # above tests use a server certificate for "localhost",
    # client connection to "localhost" too.
    # here we test that even if the server certificate is for another domain,
    # "www.example.com" in this case,
    # the tests still pass
    keyfile = 'keys/example-com.key.pem'
    certfile = 'keys/example-com.cert.pem'


class TestRequestMetaStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN}


class MixinSameOrigin(object):
    scenarii = [
        # Same origin (protocol, host, port): send referrer
        ('https://example.com/page.html',       'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('https://example.com:443/page.html',   'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com:80/page.html',     'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com:80/not-page.html',      b'http://example.com/page.html'),
        ('http://example.com:8888/page.html',   'http://example.com:8888/not-page.html',    b'http://example.com:8888/page.html'),

        # Different host: do NOT send referrer
        ('https://example.com/page.html',       'https://not.example.com/otherpage.html',   None),
        ('http://example.com/page.html',        'http://not.example.com/otherpage.html',    None),
        ('http://example.com/page.html',        'http://www.example.com/otherpage.html',    None),

        # Different port: do NOT send referrer
        ('https://example.com:444/page.html',   'https://example.com/not-page.html',    None),
        ('http://example.com:81/page.html',     'http://example.com/not-page.html',     None),
        ('http://example.com/page.html',        'http://example.com:81/not-page.html',  None),

        # Different protocols: do NOT send refferer
        ('https://example.com/page.html',   'http://example.com/not-page.html',     None),
        ('https://example.com/page.html',   'http://not.example.com/',              None),
        ('ftps://example.com/urls.zip',     'https://example.com/not-page.html',    None),
        ('ftp://example.com/urls.zip',      'http://example.com/not-page.html',     None),
        ('ftps://example.com/urls.zip',     'https://example.com/not-page.html',    None),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'https://example.com/not-page.html',    b'https://example.com/page.html'),
        ('https://user:password@example.com/page.html', 'http://example.com/not-page.html',     None),
    ]


class MemoizedMethodTest(unittest.TestCase):
    def test_memoizemethod_noargs(self):
        class A(object):

            @memoizemethod_noargs
            def cached(self):
                return object()

            def noncached(self):
                return object()

        a = A()
        one = a.cached()
        two = a.cached()
        three = a.noncached()
        assert one is two
        assert one is not three


class Http10TestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HTTP10DownloadHandler


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


class SendCatchLogDeferredTest(SendCatchLogTest):

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)


class MySpider2(MyBaseSpider):
    name = 'myspider2'

class TestDownloaderStats(TestCase):

    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('scrapytest.org')
        self.mw = DownloaderStats(self.crawler.stats)

        self.crawler.stats.open_spider(self.spider)

        self.req = Request('http://scrapytest.org')
        self.res = Response('scrapytest.org', status=400)

    def assertStatsEqual(self, key, value):
        self.assertEqual(
            self.crawler.stats.get_value(key, spider=self.spider),
            value,
            str(self.crawler.stats.get_stats(self.spider))
        )

    def test_process_request(self):
        self.mw.process_request(self.req, self.spider)
        self.assertStatsEqual('downloader/request_count', 1)

    def test_process_response(self):
        self.mw.process_response(self.req, self.res, self.spider)
        self.assertStatsEqual('downloader/response_count', 1)

    def test_process_exception(self):
        self.mw.process_exception(self.req, MyException(), self.spider)
        self.assertStatsEqual('downloader/exception_count', 1)
        self.assertStatsEqual(
            'downloader/exception_type_count/tests.test_downloadermiddleware_stats.MyException',
            1
        )

    def tearDown(self):
        self.crawler.stats.close_spider(self.spider, '')

class TestSettingsCustomPolicy(TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'tests.test_spidermiddleware_referer.CustomPythonOrgPolicy'}
    scenarii = [
        ('https://example.com/',    'https://scrapy.org/',  b'https://python.org/'),
        ('http://example.com/',     'http://scrapy.org/',   b'http://python.org/'),
        ('http://example.com/',     'https://scrapy.org/',  b'https://python.org/'),
        ('https://example.com/',    'http://scrapy.org/',   b'http://python.org/'),
        ('file:///home/path/to/somefile.html',  'https://scrapy.org/', b'https://python.org/'),
        ('file:///home/path/to/somefile.html',  'http://scrapy.org/',  b'http://python.org/'),

    ]

# --- Tests using Request meta dict to set policy
class LogformatterSubclassTest(LoggingContribTest):
    def setUp(self):
        self.formatter = LogFormatterSubclass()
        self.spider = Spider('default')

    def test_flags_in_request(self):
        pass


if __name__ == "__main__":
    unittest.main()

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

class TestSpider(Spider):
    name = "scrapytest.org"
    allowed_domains = ["scrapytest.org", "localhost"]

    itemurl_re = re.compile("item\d+.html")
    name_re = re.compile("<h1>(.*?)</h1>", re.M)
    price_re = re.compile(">Price: \$(.*?)<", re.M)

    item_cls = TestItem

    def parse(self, response):
        xlink = LinkExtractor()
        itemre = re.compile(self.itemurl_re)
        for link in xlink.extract_links(response):
            if itemre.search(link.url):
                yield Request(url=link.url, callback=self.parse_item)

    def parse_item(self, response):
        item = self.item_cls()
        m = self.name_re.search(response.text)
        if m:
            item['name'] = m.group(1)
        item['url'] = response.url
        m = self.price_re.search(response.text)
        if m:
            item['price'] = m.group(1)
        return item

