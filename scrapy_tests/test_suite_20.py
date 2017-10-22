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


class UtilsConfTestCase(unittest.TestCase):

    def test_arglist_to_dict(self):
        self.assertEqual(arglist_to_dict(['arg1=val1', 'arg2=val2']),
            {'arg1': 'val1', 'arg2': 'val2'})


if __name__ == "__main__":
    unittest.main()

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

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['http://localhost:12345']

    def parse(self, response):
        return {'test': 'value'}
""",
                           args=('-s', 'DNSCACHE_ENABLED=False'))
        print(log)
        self.assertNotIn("DNSLookupError", log)
        self.assertIn("INFO: Spider opened", log)

    def test_runspider_log_short_names(self):
        log1 = self.get_log(self.debug_log_spider,
                            args=('-s', 'LOG_SHORT_NAMES=1'))
        print(log1)
        self.assertIn("[myspider] DEBUG: It Works!", log1)
        self.assertIn("[scrapy]", log1)
        self.assertNotIn("[scrapy.core.engine]", log1)

        log2 = self.get_log(self.debug_log_spider,
                            args=('-s', 'LOG_SHORT_NAMES=0'))
        print(log2)
        self.assertIn("[myspider] DEBUG: It Works!", log2)
        self.assertNotIn("[scrapy]", log2)
        self.assertIn("[scrapy.core.engine]", log2)

    def test_runspider_no_spider_found(self):
        log = self.get_log("from scrapy.spiders import Spider\n")
        self.assertIn("No spider found in file", log)

    def test_runspider_file_not_found(self):
        p = self.proc('runspider', 'some_non_existent_file')
        log = to_native_str(p.stderr.read())
        self.assertIn("File not found: some_non_existent_file", log)

    def test_runspider_unable_to_load(self):
        log = self.get_log('', name='myspider.txt')
        self.assertIn('Unable to load', log)

    def test_start_requests_errors(self):
        log = self.get_log("""
import scrapy

class MySpider(scrapy.Spider):
    name = 'myspider'

    def start_requests(self):
        self.logger.debug("It Works!")
        return []
"""

    @contextmanager
    def _create_file(self, content, name):
        tmpdir = self.mktemp()
        os.mkdir(tmpdir)
        fname = abspath(join(tmpdir, name))
        with open(fname, 'w') as f:
            f.write(content)
        try:
            yield fname
        finally:
            rmtree(tmpdir)

    def runspider(self, code, name='myspider.py', args=()):
        with self._create_file(code, name) as fname:
            return self.proc('runspider', fname, *args)

    def get_log(self, code, name='myspider.py', args=()):
        p = self.runspider(code, name=name, args=args)
        return to_native_str(p.stderr.read())

    def test_runspider(self):
        log = self.get_log(self.debug_log_spider)
        self.assertIn("DEBUG: It Works!", log)
        self.assertIn("INFO: Spider opened", log)
        self.assertIn("INFO: Closing spider (finished)", log)
        self.assertIn("INFO: Spider closed (finished)", log)

    def test_runspider_log_level(self):
        log = self.get_log(self.debug_log_spider,
                           args=('-s', 'LOG_LEVEL=INFO'))
        self.assertNotIn("DEBUG: It Works!", log)
        self.assertIn("INFO: Spider opened", log)

    def test_runspider_dnscache_disabled(self):
        # see https://github.com/scrapy/scrapy/issues/2811
        # The spider below should not be able to connect to localhost:12345,
        # which is intended,
        # but this should not be because of DNS lookup error
        # assumption: localhost will resolve in all cases (true?)
        log = self.get_log("""
import scrapy

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



class TestPolicyHeaderPredecence003(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER_WHEN_DOWNGRADE.title()}

class _HttpErrorSpider(Spider):
    name = 'httperror'
    start_urls = [
        "http://localhost:8998/status?n=200",
        "http://localhost:8998/status?n=404",
        "http://localhost:8998/status?n=402",
        "http://localhost:8998/status?n=500",
    ]
    bypass_status_codes = set()

    def __init__(self, *args, **kwargs):
        super(_HttpErrorSpider, self).__init__(*args, **kwargs)
        self.failed = set()
        self.skipped = set()
        self.parsed = set()

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, self.parse, errback=self.on_error)

    def parse(self, response):
        self.parsed.add(response.url[-3:])

    def on_error(self, failure):
        if isinstance(failure.value, HttpError):
            response = failure.value.response
            if response.status in self.bypass_status_codes:
                self.skipped.add(response.url[-3:])
                return self.parse(response)

        # it assumes there is a response attached to failure
        self.failed.add(failure.value.response.url[-3:])
        return failure


def _responses(request, status_codes):
    responses = []
    for code in status_codes:
        response = Response(request.url, status=code)
        response.request = request
        responses.append(response)
    return responses


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

class MaxRetryTimesTest(unittest.TestCase):
    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')
        self.mw = RetryMiddleware.from_crawler(self.crawler)
        self.mw.max_retry_times = 2
        self.invalid_url = 'http://www.scrapytest.org/invalid_url'

    def test_with_settings_zero(self):

        # SETTINGS: RETRY_TIMES = 0
        self.mw.max_retry_times = 0

        req = Request(self.invalid_url)
        self._test_retry(req, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_zero(self):

        # SETTINGS: meta(max_retry_times) = 0
        meta_max_retry_times = 0
        
        req = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        self._test_retry(req, DNSLookupError('foo'), meta_max_retry_times)

    def test_without_metakey(self):

        # SETTINGS: RETRY_TIMES is NON-ZERO
        self.mw.max_retry_times = 5

        req = Request(self.invalid_url)
        self._test_retry(req, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_greater(self):
        
        # SETINGS: RETRY_TIMES < meta(max_retry_times)
        self.mw.max_retry_times = 2
        meta_max_retry_times = 3

        req1 = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        req2 = Request(self.invalid_url)

        self._test_retry(req1, DNSLookupError('foo'), meta_max_retry_times)
        self._test_retry(req2, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_lesser(self):
        
        # SETINGS: RETRY_TIMES > meta(max_retry_times)
        self.mw.max_retry_times = 5
        meta_max_retry_times = 4

        req1 = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        req2 = Request(self.invalid_url)

        self._test_retry(req1, DNSLookupError('foo'), meta_max_retry_times)
        self._test_retry(req2, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_dont_retry(self):

        # SETTINGS: meta(max_retry_times) = 4
        meta_max_retry_times = 4

        req = Request(self.invalid_url, meta= \
            {'max_retry_times': meta_max_retry_times, 'dont_retry': True})

        self._test_retry(req, DNSLookupError('foo'), 0)


    def _test_retry(self, req, exception, max_retry_times):
        
        for i in range(0, max_retry_times):
            req = self.mw.process_exception(req, exception, self.spider)
            assert isinstance(req, Request)

        # discard it
        req = self.mw.process_exception(req, exception, self.spider)
        self.assertEqual(req, None)


if __name__ == "__main__":
    unittest.main()

class TestSettingsSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}


class _BaseTest(unittest.TestCase):

    storage_class = 'scrapy.extensions.httpcache.DbmCacheStorage'
    policy_class = 'scrapy.extensions.httpcache.RFC2616Policy'

    def setUp(self):
        self.yesterday = email.utils.formatdate(time.time() - 86400)
        self.today = email.utils.formatdate()
        self.tomorrow = email.utils.formatdate(time.time() + 86400)
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('example.com')
        self.tmpdir = tempfile.mkdtemp()
        self.request = Request('http://www.example.com',
                               headers={'User-Agent': 'test'})
        self.response = Response('http://www.example.com',
                                 headers={'Content-Type': 'text/html'},
                                 body=b'test body',
                                 status=202)
        self.crawler.stats.open_spider(self.spider)

    def tearDown(self):
        self.crawler.stats.close_spider(self.spider, '')
        shutil.rmtree(self.tmpdir)

    def _get_settings(self, **new_settings):
        settings = {
            'HTTPCACHE_ENABLED': True,
            'HTTPCACHE_DIR': self.tmpdir,
            'HTTPCACHE_EXPIRATION_SECS': 1,
            'HTTPCACHE_IGNORE_HTTP_CODES': [],
            'HTTPCACHE_POLICY': self.policy_class,
            'HTTPCACHE_STORAGE': self.storage_class,
        }
        settings.update(new_settings)
        return Settings(settings)

    @contextmanager
    def _storage(self, **new_settings):
        with self._middleware(**new_settings) as mw:
            yield mw.storage

    @contextmanager
    def _policy(self, **new_settings):
        with self._middleware(**new_settings) as mw:
            yield mw.policy

    @contextmanager
    def _middleware(self, **new_settings):
        settings = self._get_settings(**new_settings)
        mw = HttpCacheMiddleware(settings, self.crawler.stats)
        mw.spider_opened(self.spider)
        try:
            yield mw
        finally:
            mw.spider_closed(self.spider)

    def assertEqualResponse(self, response1, response2):
        self.assertEqual(response1.url, response2.url)
        self.assertEqual(response1.status, response2.status)
        self.assertEqual(response1.headers, response2.headers)
        self.assertEqual(response1.body, response2.body)

    def assertEqualRequest(self, request1, request2):
        self.assertEqual(request1.url, request2.url)
        self.assertEqual(request1.headers, request2.headers)
        self.assertEqual(request1.body, request2.body)

    def assertEqualRequestButWithCacheValidators(self, request1, request2):
        self.assertEqual(request1.url, request2.url)
        assert not b'If-None-Match' in request1.headers
        assert not b'If-Modified-Since' in request1.headers
        assert any(h in request2.headers for h in (b'If-None-Match', b'If-Modified-Since'))
        self.assertEqual(request1.body, request2.body)

    def test_dont_cache(self):
        with self._middleware() as mw:
            self.request.meta['dont_cache'] = True
            mw.process_response(self.request, self.response, self.spider)
            self.assertEqual(mw.storage.retrieve_response(self.spider, self.request), None)

        with self._middleware() as mw:
            self.request.meta['dont_cache'] = False
            mw.process_response(self.request, self.response, self.spider)
            if mw.policy.should_cache_response(self.response, self.request):
                self.assertIsInstance(mw.storage.retrieve_response(self.spider, self.request), self.response.__class__)


class TestOffsiteMiddleware4(TestOffsiteMiddleware3):

    def _get_spider(self):
      bad_hostname = urlparse('http:////scrapytest.org').hostname
      return dict(name='foo', allowed_domains=['scrapytest.org', None, bad_hostname])

    def test_process_spider_output(self):
      res = Response('http://scrapytest.org')
      reqs = [Request('http://scrapytest.org/1')]
      out = list(self.mw.process_spider_output(res, reqs, self.spider))
      self.assertEqual(out, reqs)

class LxmlXmliterTestCase(XmliterTestCase):
    xmliter = staticmethod(xmliter_lxml)

    def test_xmliter_iterate_namespace(self):
        body = b"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0" xmlns="http://base.google.com/ns/1.0">
                <channel>
                <title>My Dummy Company</title>
                <link>http://www.mydummycompany.com</link>
                <description>This is a dummy company. We do nothing.</description>
                <item>
                    <title>Item 1</title>
                    <description>This is item 1</description>
                    <link>http://www.mydummycompany.com/items/1</link>
                    <image_link>http://www.mydummycompany.com/images/item1.jpg</image_link>
                    <image_link>http://www.mydummycompany.com/images/item2.jpg</image_link>
                </item>
                </channel>
            </rss>
        """
        response = XmlResponse(url='http://mydummycompany.com', body=body)

        no_namespace_iter = self.xmliter(response, 'image_link')
        self.assertEqual(len(list(no_namespace_iter)), 0)

        namespace_iter = self.xmliter(response, 'image_link', 'http://base.google.com/ns/1.0')
        node = next(namespace_iter)
        self.assertEqual(node.xpath('text()').extract(), ['http://www.mydummycompany.com/images/item1.jpg'])
        node = next(namespace_iter)
        self.assertEqual(node.xpath('text()').extract(), ['http://www.mydummycompany.com/images/item2.jpg'])

    def test_xmliter_namespaces_prefix(self):
        body = b"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <root>
            <h:table xmlns:h="http://www.w3.org/TR/html4/">
              <h:tr>
                <h:td>Apples</h:td>
                <h:td>Bananas</h:td>
              </h:tr>
            </h:table>

            <f:table xmlns:f="http://www.w3schools.com/furniture">
              <f:name>African Coffee Table</f:name>
              <f:width>80</f:width>
              <f:length>120</f:length>
            </f:table>

        </root>
        """
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'table', 'http://www.w3.org/TR/html4/', 'h')

        node = next(my_iter)
        self.assertEqual(len(node.xpath('h:tr/h:td').extract()), 2)
        self.assertEqual(node.xpath('h:tr/h:td[1]/text()').extract(), ['Apples'])
        self.assertEqual(node.xpath('h:tr/h:td[2]/text()').extract(), ['Bananas'])

        my_iter = self.xmliter(response, 'table', 'http://www.w3schools.com/furniture', 'f')

        node = next(my_iter)
        self.assertEqual(node.xpath('f:name/text()').extract(), ['African Coffee Table'])

    def test_xmliter_objtype_exception(self):
        i = self.xmliter(42, 'product')
        self.assertRaises(TypeError, next, i)

class MyGoodCrawlSpider(CrawlSpider):
    name = 'goodcrawl{0}'

    rules = (
        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'/text'), follow=True),
    )

    def parse_item(self, response):
        return [scrapy.Item(), dict(foo='bar')]

    def parse(self, response):
        return [scrapy.Item(), dict(nomatch='default')]


class FeedExportTest(unittest.TestCase):

    class MyItem(scrapy.Item):
        foo = scrapy.Field()
        egg = scrapy.Field()
        baz = scrapy.Field()

    @defer.inlineCallbacks
    def run_and_export(self, spider_cls, settings=None):
        """ Run spider with specified settings; return exported data. """
        tmpdir = tempfile.mkdtemp()
        res_name = tmpdir + '/res'
        defaults = {
            'FEED_URI': 'file://' + res_name,
            'FEED_FORMAT': 'csv',
        }
        defaults.update(settings or {})
        try:
            with MockServer() as s:
                runner = CrawlerRunner(Settings(defaults))
                yield runner.crawl(spider_cls)

            with open(res_name, 'rb') as f:
                defer.returnValue(f.read())

        finally:
            shutil.rmtree(tmpdir)

    @defer.inlineCallbacks
    def exported_data(self, items, settings):
        """
        Return exported data which a spider yielding ``items`` would return.
        """
        class TestSpider(scrapy.Spider):
            name = 'testspider'
            start_urls = ['http://localhost:8998/']

            def parse(self, response):
                for item in items:
                    yield item

        data = yield self.run_and_export(TestSpider, settings)
        defer.returnValue(data)

    @defer.inlineCallbacks
    def exported_no_data(self, settings):
        """
        Return exported data which a spider yielding no ``items`` would return.
        """
        class TestSpider(scrapy.Spider):
            name = 'testspider'
            start_urls = ['http://localhost:8998/']

            def parse(self, response):
                pass

        data = yield self.run_and_export(TestSpider, settings)
        defer.returnValue(data)

    @defer.inlineCallbacks
    def assertExportedCsv(self, items, header, rows, settings=None, ordered=True):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'csv'})
        data = yield self.exported_data(items, settings)

        reader = csv.DictReader(to_native_str(data).splitlines())
        got_rows = list(reader)
        if ordered:
            self.assertEqual(reader.fieldnames, header)
        else:
            self.assertEqual(set(reader.fieldnames), set(header))

        self.assertEqual(rows, got_rows)

    @defer.inlineCallbacks
    def assertExportedJsonLines(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'jl'})
        data = yield self.exported_data(items, settings)
        parsed = [json.loads(to_native_str(line)) for line in data.splitlines()]
        rows = [{k: v for k, v in row.items() if v} for row in rows]
        self.assertEqual(rows, parsed)

    @defer.inlineCallbacks
    def assertExportedXml(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'xml'})
        data = yield self.exported_data(items, settings)
        rows = [{k: v for k, v in row.items() if v} for row in rows]
        import lxml.etree
        root = lxml.etree.fromstring(data)
        got_rows = [{e.tag: e.text for e in it} for it in root.findall('item')]
        self.assertEqual(rows, got_rows)

    def _load_until_eof(self, data, load_func):
        bytes_output = BytesIO(data)
        result = []
        while True:
            try:
                result.append(load_func(bytes_output))
            except EOFError:
                break
        return result

    @defer.inlineCallbacks
    def assertExportedPickle(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'pickle'})
        data = yield self.exported_data(items, settings)
        expected = [{k: v for k, v in row.items() if v} for row in rows]
        import pickle
        result = self._load_until_eof(data, load_func=pickle.load)
        self.assertEqual(expected, result)

    @defer.inlineCallbacks
    def assertExportedMarshal(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'marshal'})
        data = yield self.exported_data(items, settings)
        expected = [{k: v for k, v in row.items() if v} for row in rows]
        import marshal
        result = self._load_until_eof(data, load_func=marshal.load)
        self.assertEqual(expected, result)

    @defer.inlineCallbacks
    def assertExported(self, items, header, rows, settings=None, ordered=True):
        yield self.assertExportedCsv(items, header, rows, settings, ordered)
        yield self.assertExportedJsonLines(items, rows, settings)
        yield self.assertExportedXml(items, rows, settings)
        yield self.assertExportedPickle(items, rows, settings)

    @defer.inlineCallbacks
    def test_export_items(self):
        # feed exporters use field names from Item
        items = [
            self.MyItem({'foo': 'bar1', 'egg': 'spam1'}),
            self.MyItem({'foo': 'bar2', 'egg': 'spam2', 'baz': 'quux2'}),
        ]
        rows = [
            {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
            {'egg': 'spam2', 'foo': 'bar2', 'baz': 'quux2'}
        ]
        header = self.MyItem.fields.keys()
        yield self.assertExported(items, header, rows, ordered=False)

    @defer.inlineCallbacks
    def test_export_no_items_not_store_empty(self):
        formats = ('json',
                   'jsonlines',
                   'xml',
                   'csv',)

        for fmt in formats:
            settings = {'FEED_FORMAT': fmt}
            data = yield self.exported_no_data(settings)
            self.assertEqual(data, b'')

    @defer.inlineCallbacks
    def test_export_no_items_store_empty(self):
        formats = (
            ('json', b'[]'),
            ('jsonlines', b''),
            ('xml', b'<?xml version="1.0" encoding="utf-8"?>\n<items></items>'),
            ('csv', b''),
        )

        for fmt, expctd in formats:
            settings = {'FEED_FORMAT': fmt, 'FEED_STORE_EMPTY': True, 'FEED_EXPORT_INDENT': None}
            data = yield self.exported_no_data(settings)
            self.assertEqual(data, expctd)

    @defer.inlineCallbacks
    def test_export_multiple_item_classes(self):

        class MyItem2(scrapy.Item):
            foo = scrapy.Field()
            hello = scrapy.Field()

        items = [
            self.MyItem({'foo': 'bar1', 'egg': 'spam1'}),
            MyItem2({'hello': 'world2', 'foo': 'bar2'}),
            self.MyItem({'foo': 'bar3', 'egg': 'spam3', 'baz': 'quux3'}),
            {'hello': 'world4', 'egg': 'spam4'},
        ]

        # by default, Scrapy uses fields of the first Item for CSV and
        # all fields for JSON Lines
        header = self.MyItem.fields.keys()
        rows_csv = [
            {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
            {'egg': '',      'foo': 'bar2', 'baz': ''},
            {'egg': 'spam3', 'foo': 'bar3', 'baz': 'quux3'},
            {'egg': 'spam4', 'foo': '',     'baz': ''},
        ]
        rows_jl = [dict(row) for row in items]
        yield self.assertExportedCsv(items, header, rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl)

        # edge case: FEED_EXPORT_FIELDS==[] means the same as default None
        settings = {'FEED_EXPORT_FIELDS': []}
        yield self.assertExportedCsv(items, header, rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl, settings)

        # it is possible to override fields using FEED_EXPORT_FIELDS
        header = ["foo", "baz", "hello"]
        settings = {'FEED_EXPORT_FIELDS': header}
        rows = [
            {'foo': 'bar1', 'baz': '',      'hello': ''},
            {'foo': 'bar2', 'baz': '',      'hello': 'world2'},
            {'foo': 'bar3', 'baz': 'quux3', 'hello': ''},
            {'foo': '',     'baz': '',      'hello': 'world4'},
        ]
        yield self.assertExported(items, header, rows,
                                  settings=settings, ordered=True)

    @defer.inlineCallbacks
    def test_export_dicts(self):
        # When dicts are used, only keys from the first row are used as
        # a header for CSV, and all fields are used for JSON Lines.
        items = [
            {'foo': 'bar', 'egg': 'spam'},
            {'foo': 'bar', 'egg': 'spam', 'baz': 'quux'},
        ]
        rows_csv = [
            {'egg': 'spam', 'foo': 'bar'},
            {'egg': 'spam', 'foo': 'bar'}
        ]
        rows_jl = items
        yield self.assertExportedCsv(items, ['egg', 'foo'], rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl)

    @defer.inlineCallbacks
    def test_export_feed_export_fields(self):
        # FEED_EXPORT_FIELDS option allows to order export fields
        # and to select a subset of fields to export, both for Items and dicts.

        for item_cls in [self.MyItem, dict]:
            items = [
                item_cls({'foo': 'bar1', 'egg': 'spam1'}),
                item_cls({'foo': 'bar2', 'egg': 'spam2', 'baz': 'quux2'}),
            ]

            # export all columns
            settings = {'FEED_EXPORT_FIELDS': 'foo,baz,egg'}
            rows = [
                {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
                {'egg': 'spam2', 'foo': 'bar2', 'baz': 'quux2'}
            ]
            yield self.assertExported(items, ['foo', 'baz', 'egg'], rows,
                                      settings=settings, ordered=True)

            # export a subset of columns
            settings = {'FEED_EXPORT_FIELDS': 'egg,baz'}
            rows = [
                {'egg': 'spam1', 'baz': ''},
                {'egg': 'spam2', 'baz': 'quux2'}
            ]
            yield self.assertExported(items, ['egg', 'baz'], rows,
                                      settings=settings, ordered=True)

    @defer.inlineCallbacks
    def test_export_encoding(self):
        items = [dict({'foo': u'Test\xd6'})]
        header = ['foo']

        formats = {
            'json': u'[{"foo": "Test\\u00d6"}]'.encode('utf-8'),
            'jsonlines': u'{"foo": "Test\\u00d6"}\n'.encode('utf-8'),
            'xml': u'<?xml version="1.0" encoding="utf-8"?>\n<items><item><foo>Test\xd6</foo></item></items>'.encode('utf-8'),
            'csv': u'foo\r\nTest\xd6\r\n'.encode('utf-8'),
        }

        for format, expected in formats.items():
            settings = {'FEED_FORMAT': format, 'FEED_EXPORT_INDENT': None}
            data = yield self.exported_data(items, settings)
            self.assertEqual(expected, data)

        formats = {
            'json': u'[{"foo": "Test\xd6"}]'.encode('latin-1'),
            'jsonlines': u'{"foo": "Test\xd6"}\n'.encode('latin-1'),
            'xml': u'<?xml version="1.0" encoding="latin-1"?>\n<items><item><foo>Test\xd6</foo></item></items>'.encode('latin-1'),
            'csv': u'foo\r\nTest\xd6\r\n'.encode('latin-1'),
        }

        settings = {'FEED_EXPORT_INDENT': None, 'FEED_EXPORT_ENCODING': 'latin-1'}
        for format, expected in formats.items():
            settings['FEED_FORMAT'] = format
            data = yield self.exported_data(items, settings)
            self.assertEqual(expected, data)

    @defer.inlineCallbacks
    def test_export_indentation(self):
        items = [
            {'foo': ['bar']},
            {'key': 'value'},
        ]

        test_cases = [
            # JSON
            {
                'format': 'json',
                'indent': None,
                'expected': b'[{"foo": ["bar"]},{"key": "value"}]',
            },
            {
                'format': 'json',
                'indent': -1,
                'expected': b"""[
{"foo": ["bar"]},
{"key": "value"}
]""",
            },
            {
                'format': 'json',
                'indent': 0,
                'expected': b"""[
{"foo": ["bar"]},
{"key": "value"}
]""",
            },
            {
                'format': 'json',
                'indent': 2,
                'expected': b"""[
{
  "foo": [
    "bar"
  ]
},
{
  "key": "value"
}
]""",
            },
            {
                'format': 'json',
                'indent': 4,
                'expected': b"""[
{
    "foo": [
        "bar"
    ]
},
{
    "key": "value"
}
]""",
            },
            {
                'format': 'json',
                'indent': 5,
                'expected': b"""[
{
     "foo": [
          "bar"
     ]
},
{
     "key": "value"
}
]""",
            },

            # XML
            {
                'format': 'xml',
                'indent': None,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items><item><foo><value>bar</value></foo></item><item><key>value</key></item></items>""",
            },
            {
                'format': 'xml',
                'indent': -1,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
<item><foo><value>bar</value></foo></item>
<item><key>value</key></item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 0,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
<item><foo><value>bar</value></foo></item>
<item><key>value</key></item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 2,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
  <item>
    <foo>
      <value>bar</value>
    </foo>
  </item>
  <item>
    <key>value</key>
  </item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 4,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
    <item>
        <foo>
            <value>bar</value>
        </foo>
    </item>
    <item>
        <key>value</key>
    </item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 5,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
     <item>
          <foo>
               <value>bar</value>
          </foo>
     </item>
     <item>
          <key>value</key>
     </item>
</items>""",
            },
        ]

        for row in test_cases:
            settings = {'FEED_FORMAT': row['format'], 'FEED_EXPORT_INDENT': row['indent']}
            data = yield self.exported_data(items, settings)
            print(row['format'], row['indent'])
            self.assertEqual(row['expected'], data)

class ItemMetaClassCellRegression(unittest.TestCase):

    def test_item_meta_classcell_regression(self):
        class MyItem(six.with_metaclass(ItemMeta, Item)):
            def __init__(self, *args, **kwargs):
                # This call to super() trigger the __classcell__ propagation
                # requirement. When not done properly raises an error:
                # TypeError: __class__ set to <class '__main__.MyItem'>
                # defining 'MyItem' as <class '__main__.MyItem'>
                super(MyItem, self).__init__(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()

class RFPDupeFilterTest(unittest.TestCase):

    def test_filter(self):
        dupefilter = RFPDupeFilter()
        dupefilter.open()

        r1 = Request('http://scrapytest.org/1')
        r2 = Request('http://scrapytest.org/2')
        r3 = Request('http://scrapytest.org/2')

        assert not dupefilter.request_seen(r1)
        assert dupefilter.request_seen(r1)

        assert not dupefilter.request_seen(r2)
        assert dupefilter.request_seen(r3)

        dupefilter.close('finished')

    def test_dupefilter_path(self):
        r1 = Request('http://scrapytest.org/1')
        r2 = Request('http://scrapytest.org/2')

        path = tempfile.mkdtemp()
        try:
            df = RFPDupeFilter(path)
            df.open()
            assert not df.request_seen(r1)
            assert df.request_seen(r1)
            df.close('finished')

            df2 = RFPDupeFilter(path)
            df2.open()
            assert df2.request_seen(r1)
            assert not df2.request_seen(r2)
            assert df2.request_seen(r2)
            df2.close('finished')
        finally:
            shutil.rmtree(path)

    def test_request_fingerprint(self):
        """Test if customization of request_fingerprint method will change
        output of request_seen.

        """
        r1 = Request('http://scrapytest.org/index.html')
        r2 = Request('http://scrapytest.org/INDEX.html')

        dupefilter = RFPDupeFilter()
        dupefilter.open()

        assert not dupefilter.request_seen(r1)
        assert not dupefilter.request_seen(r2)

        dupefilter.close('finished')

        class CaseInsensitiveRFPDupeFilter(RFPDupeFilter):

            def request_fingerprint(self, request):
                fp = hashlib.sha1()
                fp.update(to_bytes(request.url.lower()))
                return fp.hexdigest()

        case_insensitive_dupefilter = CaseInsensitiveRFPDupeFilter()
        case_insensitive_dupefilter.open()

        assert not case_insensitive_dupefilter.request_seen(r1)
        assert case_insensitive_dupefilter.request_seen(r2)

        case_insensitive_dupefilter.close('finished')

class ChunkSize2MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 2

class Bar(trackref.object_ref):
    pass


class TestItem(Item):
    name = Field()
    url = Field()


class CSVFeedSpiderTest(SpiderTest):

    spider_class = CSVFeedSpider


class TestRequestMetaPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    req_meta = {'referrer_policy': POLICY_NO_REFERRER}


class Http11ProxyTestCase(HttpProxyTestCase):
    download_handler_cls = HTTP11DownloadHandler

    @defer.inlineCallbacks
    def test_download_with_proxy_https_timeout(self):
        """ Test TunnelingTCP4ClientEndpoint """
        http_proxy = self.getURL('')
        domain = 'https://no-such-domain.nosuch'
        request = Request(
            domain, meta={'proxy': http_proxy, 'download_timeout': 0.2})
        d = self.download_request(request, Spider('foo'))
        timeout = yield self.assertFailure(d, error.TimeoutError)
        self.assertIn(domain, timeout.osError)


class DeprecationTest(unittest.TestCase):

    def test_basespider_is_deprecated(self):
        with warnings.catch_warnings(record=True) as w:

            class MySpider1(BaseSpider):
                pass

            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, ScrapyDeprecationWarning)
            self.assertEqual(w[0].lineno, inspect.getsourcelines(MySpider1)[1])

    def test_basespider_issubclass(self):
        class MySpider2(Spider):
            pass

        class MySpider2a(MySpider2):
            pass

        class Foo(object):
            pass

        class Foo2(object_ref):
            pass

        assert issubclass(MySpider2, BaseSpider)
        assert issubclass(MySpider2a, BaseSpider)
        assert not issubclass(Foo, BaseSpider)
        assert not issubclass(Foo2, BaseSpider)

    def test_basespider_isinstance(self):
        class MySpider3(Spider):
            name = 'myspider3'

        class MySpider3a(MySpider3):
            pass

        class Foo(object):
            pass

        class Foo2(object_ref):
            pass

        assert isinstance(MySpider3(), BaseSpider)
        assert isinstance(MySpider3a(), BaseSpider)
        assert not isinstance(Foo(), BaseSpider)
        assert not isinstance(Foo2(), BaseSpider)

    def test_crawl_spider(self):
        assert issubclass(CrawlSpider, Spider)
        assert issubclass(CrawlSpider, BaseSpider)
        assert isinstance(CrawlSpider(name='foo'), Spider)
        assert isinstance(CrawlSpider(name='foo'), BaseSpider)

    def test_make_requests_from_url_deprecated(self):
        class MySpider4(Spider):
            name = 'spider1'
            start_urls = ['http://example.com']

        class MySpider5(Spider):
            name = 'spider2'
            start_urls = ['http://example.com']

            def make_requests_from_url(self, url):
                return Request(url + "/foo", dont_filter=True)

        with warnings.catch_warnings(record=True) as w:
            # spider without overridden make_requests_from_url method
            # doesn't issue a warning
            spider1 = MySpider4()
            self.assertEqual(len(list(spider1.start_requests())), 1)
            self.assertEqual(len(w), 0)

            # spider with overridden make_requests_from_url issues a warning,
            # but the method still works
            spider2 = MySpider5()
            requests = list(spider2.start_requests())
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].url, 'http://example.com/foo')
            self.assertEqual(len(w), 1)


class Bar(trackref.object_ref):
    pass


class WebClientTestCase(unittest.TestCase):
    def _listen(self, site):
        return reactor.listenTCP(0, site, interface="127.0.0.1")

    def setUp(self):
        self.tmpname = self.mktemp()
        os.mkdir(self.tmpname)
        FilePath(self.tmpname).child("file").setContent(b"0123456789")
        r = static.File(self.tmpname)
        r.putChild(b"redirect", util.Redirect(b"/file"))
        r.putChild(b"wait", ForeverTakingResource())
        r.putChild(b"error", ErrorResource())
        r.putChild(b"nolength", NoLengthResource())
        r.putChild(b"host", HostHeaderResource())
        r.putChild(b"payload", PayloadResource())
        r.putChild(b"broken", BrokenDownloadResource())
        r.putChild(b"encoding", EncodingResource())
        self.site = server.Site(r, timeout=None)
        self.wrapper = WrappingFactory(self.site)
        self.port = self._listen(self.wrapper)
        self.portno = self.port.getHost().port

    @inlineCallbacks
    def tearDown(self):
        yield self.port.stopListening()
        shutil.rmtree(self.tmpname)

    def getURL(self, path):
        return "http://127.0.0.1:%d/%s" % (self.portno, path)

    def testPayload(self):
        s = "0123456789" * 10
        return getPage(self.getURL("payload"), body=s).addCallback(
            self.assertEqual, to_bytes(s))

    def testHostHeader(self):
        # if we pass Host header explicitly, it should be used, otherwise
        # it should extract from url
        return defer.gatherResults([
            getPage(self.getURL("host")).addCallback(
                self.assertEqual, to_bytes("127.0.0.1:%d" % self.portno)),
            getPage(self.getURL("host"), headers={"Host": "www.example.com"}).addCallback(
                self.assertEqual, to_bytes("www.example.com"))])

    def test_getPage(self):
        """
        L{client.getPage} returns a L{Deferred} which is called back with
        the body of the response if the default method B{GET} is used.
        """
        d = getPage(self.getURL("file"))
        d.addCallback(self.assertEqual, b"0123456789")
        return d

    def test_getPageHead(self):
        """
        L{client.getPage} returns a L{Deferred} which is called back with
        the empty string if the method is C{HEAD} and there is a successful
        response code.
        """
        def _getPage(method):
            return getPage(self.getURL("file"), method=method)
        return defer.gatherResults([
            _getPage("head").addCallback(self.assertEqual, b""),
            _getPage("HEAD").addCallback(self.assertEqual, b"")])

    def test_timeoutNotTriggering(self):
        """
        When a non-zero timeout is passed to L{getPage} and the page is
        retrieved before the timeout period elapses, the L{Deferred} is
        called back with the contents of the page.
        """
        d = getPage(self.getURL("host"), timeout=100)
        d.addCallback(
            self.assertEqual, to_bytes("127.0.0.1:%d" % self.portno))
        return d

    def test_timeoutTriggering(self):
        """
        When a non-zero timeout is passed to L{getPage} and that many
        seconds elapse before the server responds to the request. the
        L{Deferred} is errbacked with a L{error.TimeoutError}.
        """
        finished = self.assertFailure(
            getPage(self.getURL("wait"), timeout=0.000001),
            defer.TimeoutError)
        def cleanup(passthrough):
            # Clean up the server which is hanging around not doing
            # anything.
            connected = list(six.iterkeys(self.wrapper.protocols))
            # There might be nothing here if the server managed to already see
            # that the connection was lost.
            if connected:
                connected[0].transport.loseConnection()
            return passthrough
        finished.addBoth(cleanup)
        return finished

    def testNotFound(self):
        return getPage(self.getURL('notsuchfile')).addCallback(self._cbNoSuchFile)

    def _cbNoSuchFile(self, pageData):
        self.assertIn(b'404 - No Such Resource', pageData)

    def testFactoryInfo(self):
        url = self.getURL('file')
        _, _, host, port, _ = client._parse(url)
        factory = client.ScrapyHTTPClientFactory(Request(url))
        reactor.connectTCP(to_unicode(host), port, factory)
        return factory.deferred.addCallback(self._cbFactoryInfo, factory)

    def _cbFactoryInfo(self, ignoredResult, factory):
        self.assertEqual(factory.status, b'200')
        self.assertTrue(factory.version.startswith(b'HTTP/'))
        self.assertEqual(factory.message, b'OK')
        self.assertEqual(factory.response_headers[b'content-length'], b'10')

    def testRedirect(self):
        return getPage(self.getURL("redirect")).addCallback(self._cbRedirect)

    def _cbRedirect(self, pageData):
        self.assertEqual(pageData,
                b'\n<html>\n    <head>\n        <meta http-equiv="refresh" content="0;URL=/file">\n'
                b'    </head>\n    <body bgcolor="#FFFFFF" text="#000000">\n    '
                b'<a href="/file">click here</a>\n    </body>\n</html>\n')

    def test_encoding(self):
        """ Test that non-standart body encoding matches
        Content-Encoding header """
        body = b'\xd0\x81\xd1\x8e\xd0\xaf'
        return getPage(
            self.getURL('encoding'), body=body, response_transform=lambda r: r)\
            .addCallback(self._check_Encoding, body)

    def _check_Encoding(self, response, original_body):
        content_encoding = to_unicode(response.headers[b'Content-Encoding'])
        self.assertEqual(content_encoding, EncodingResource.out_encoding)
        self.assertEqual(
            response.body.decode(content_encoding), to_unicode(original_body))

class TestRequestMetaNoReferrer(MixinNoReferrer, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_NO_REFERRER}

