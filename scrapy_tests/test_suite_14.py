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


class CrawlerRunnerTestCase(BaseCrawlerTest):

    def test_spider_manager_verify_interface(self):
        settings = Settings({
            'SPIDER_LOADER_CLASS': 'tests.test_crawler.SpiderLoaderWithWrongInterface'
        })
        with warnings.catch_warnings(record=True) as w, \
                self.assertRaises(AttributeError):
            CrawlerRunner(settings)
            self.assertEqual(len(w), 1)
            self.assertIn("SPIDER_LOADER_CLASS", str(w[0].message))
            self.assertIn("scrapy.interfaces.ISpiderLoader", str(w[0].message))

    def test_crawler_runner_accepts_dict(self):
        runner = CrawlerRunner({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_runner_accepts_None(self):
        runner = CrawlerRunner()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_deprecated_attribute_spiders(self):
        with warnings.catch_warnings(record=True) as w:
            runner = CrawlerRunner(Settings())
            spiders = runner.spiders
            self.assertEqual(len(w), 1)
            self.assertIn("CrawlerRunner.spiders", str(w[0].message))
            self.assertIn("CrawlerRunner.spider_loader", str(w[0].message))
            sl_cls = load_object(runner.settings['SPIDER_LOADER_CLASS'])
            self.assertIsInstance(spiders, sl_cls)

    def test_spidermanager_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            runner = CrawlerRunner({
                'SPIDER_MANAGER_CLASS': 'tests.test_crawler.CustomSpiderLoader'
            })
            self.assertIsInstance(runner.spider_loader, CustomSpiderLoader)
            self.assertEqual(len(w), 1)
            self.assertIn('Please use SPIDER_LOADER_CLASS', str(w[0].message))


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

class TestPolicyHeaderPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER.swapcase()}

class CommandTest(ProjectTest):

    def setUp(self):
        super(CommandTest, self).setUp()
        self.call('startproject', self.project_name)
        self.cwd = join(self.temp_path, self.project_name)
        self.env['SCRAPY_SETTINGS_MODULE'] = '%s.settings' % self.project_name


class TestRequestMetaPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class NoParseMethodSpiderTest(unittest.TestCase):

    spider_class = Spider

    def test_undefined_parse_method(self):
        spider = self.spider_class('example.com')
        text = b'Random text'
        resp = TextResponse(url="http://www.example.com/random_url", body=text)

        exc_msg = 'Spider.parse callback is not defined'
        with self.assertRaisesRegexp(NotImplementedError, exc_msg):
            spider.parse(resp)

class TestCloseSpider(TestCase):

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_closespider_itemcount(self):
        close_on = 5
        crawler = get_crawler(ItemSpider, {'CLOSESPIDER_ITEMCOUNT': close_on})
        yield crawler.crawl()
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_itemcount')
        itemcount = crawler.stats.get_value('item_scraped_count')
        self.assertTrue(itemcount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_pagecount(self):
        close_on = 5
        crawler = get_crawler(FollowAllSpider, {'CLOSESPIDER_PAGECOUNT': close_on})
        yield crawler.crawl()
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_pagecount')
        pagecount = crawler.stats.get_value('response_received_count')
        self.assertTrue(pagecount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_errorcount(self):
        close_on = 5
        crawler = get_crawler(ErrorSpider, {'CLOSESPIDER_ERRORCOUNT': close_on})
        yield crawler.crawl(total=1000000)
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_errorcount')
        key = 'spider_exceptions/{name}'\
                .format(name=crawler.spider.exception_cls.__name__)
        errorcount = crawler.stats.get_value(key)
        self.assertTrue(errorcount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_timeout(self):
        close_on = 0.1
        crawler = get_crawler(FollowAllSpider, {'CLOSESPIDER_TIMEOUT': close_on})
        yield crawler.crawl(total=1000000)
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_timeout')
        stats = crawler.stats
        start = stats.get_value('start_time')
        stop = stats.get_value('finish_time')
        diff = stop - start
        total_seconds = diff.seconds + diff.microseconds
        self.assertTrue(total_seconds >= close_on)

class PickleLifoDiskQueueTest(MarshalLifoDiskQueueTest):

    def queue(self):
        return PickleLifoDiskQueue(self.qpath)

    def test_serialize_item(self):
        q = self.queue()
        i = TestItem(name='foo')
        q.push(i)
        i2 = q.pop()
        assert isinstance(i2, TestItem)
        self.assertEqual(i, i2)

    def test_serialize_loader(self):
        q = self.queue()
        l = TestLoader()
        q.push(l)
        l2 = q.pop()
        assert isinstance(l2, TestLoader)
        assert l2.default_item_class is TestItem
        self.assertEqual(l2.name_out('x'), 'xx')

    def test_serialize_request_recursive(self):
        q = self.queue()
        r = Request('http://www.example.com')
        r.meta['request'] = r
        q.push(r)
        r2 = q.pop()
        assert isinstance(r2, Request)
        self.assertEqual(r.url, r2.url)
        assert r2.meta['request'] is r2

class MarshalItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        self.output = tempfile.TemporaryFile()
        return MarshalItemExporter(self.output, **kwargs)

    def _check_output(self):
        self.output.seek(0)
        self._assert_expected_item(marshal.load(self.output))

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        item.pop('time')  # datetime is not marshallable
        fp = tempfile.TemporaryFile()
        ie = MarshalItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        fp.seek(0)
        self.assertEqual(marshal.load(fp), item)


class HtmlParserLinkExtractorTestCase(unittest.TestCase):

    def setUp(self):
        body = get_testdata('link_extractor', 'sgml_linkextractor.html')
        self.response = HtmlResponse(url='http://example.com/index', body=body)

    def test_extraction(self):
        # Default arguments
        lx = HtmlParserLinkExtractor()
        self.assertEqual(lx.extract_links(self.response), [
            Link(url='http://example.com/sample2.html', text=u'sample 2'),
            Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
            Link(url='http://example.com/sample3.html', text=u'sample 3 repetition'),
            Link(url='http://example.com/sample3.html#foo', text=u'sample 3 repetition with fragment'),
            Link(url='http://www.google.com/something', text=u''),
            Link(url='http://example.com/innertag.html', text=u'inner tag'),
            Link(url='http://example.com/page%204.html', text=u'href with whitespaces'),
        ])

    def test_link_wrong_href(self):
        html = """
        <a href="http://example.org/item1.html">Item 1</a>
        <a href="http://[example.org/item2.html">Item 2</a>
        <a href="http://example.org/item3.html">Item 3</a>
        """
        response = HtmlResponse("http://example.org/index.html", body=html)
        lx = HtmlParserLinkExtractor()
        self.assertEqual([link for link in lx.extract_links(response)], [
            Link(url='http://example.org/item1.html', text=u'Item 1', nofollow=False),
            Link(url='http://example.org/item3.html', text=u'Item 3', nofollow=False),
        ])


class DeprecatedFilesPipeline(FilesPipeline):
    def file_key(self, url):
        media_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        media_ext = os.path.splitext(url)[1]
        return 'empty/%s%s' % (media_guid, media_ext)


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

class ToUnicodeTest(unittest.TestCase):
    def test_converting_an_utf8_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xc3\xb1e'), u'lel\xf1e')

    def test_converting_a_latin_1_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xf1e', 'latin-1'), u'lel\xf1e')

    def test_converting_a_unicode_to_unicode_should_return_the_same_object(self):
        self.assertEqual(to_unicode(u'\xf1e\xf1e\xf1e'), u'\xf1e\xf1e\xf1e')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        self.assertRaises(TypeError, to_unicode, 423)

    def test_errors_argument(self):
        self.assertEqual(
            to_unicode(b'a\xedb', 'utf-8', errors='replace'),
            u'a\ufffdb'
        )


class TestSettingsNoReferrerWhenDowngrade(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}


class Https10TestCase(Http10TestCase):
    scheme = 'https'


class TestSettingsNoReferrerWhenDowngrade(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}


class SendCatchLogDeferredTest(SendCatchLogTest):

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)


class IterErrbackTest(unittest.TestCase):

    def test_iter_errback_good(self):
        def itergood():
            for x in xrange(10):
                yield x

        errors = []
        out = list(iter_errback(itergood(), errors.append))
        self.assertEqual(out, list(range(10)))
        self.assertFalse(errors)

    def test_iter_errback_bad(self):
        def iterbad():
            for x in xrange(10):
                if x == 5:
                    a = 1/0
                yield x

        errors = []
        out = list(iter_errback(iterbad(), errors.append))
        self.assertEqual(out, [0, 1, 2, 3, 4])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0].value, ZeroDivisionError)

class DeprecatedXpathSelectorTest(unittest.TestCase):

    text = '<div><img src="a.jpg"><p>Hello</div>'

    def test_warnings_xpathselector(self):
        cls = XPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))

    def test_warnings_xmlxpathselector(self):
        cls = XmlXPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(issubclass(cls, XPathSelector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))
            self.assertTrue(isinstance(sel, XPathSelector))
            self.assertTrue(isinstance(usel, XPathSelector))

    def test_warnings_htmlxpathselector(self):
        cls = HtmlXPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(issubclass(cls, XPathSelector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))
            self.assertTrue(isinstance(sel, XPathSelector))
            self.assertTrue(isinstance(usel, XPathSelector))

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

class TestRequestMetaOrigin(MixinOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN}


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

class MarshalItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        self.output = tempfile.TemporaryFile()
        return MarshalItemExporter(self.output, **kwargs)

    def _check_output(self):
        self.output.seek(0)
        self._assert_expected_item(marshal.load(self.output))

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        item.pop('time')  # datetime is not marshallable
        fp = tempfile.TemporaryFile()
        ie = MarshalItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        fp.seek(0)
        self.assertEqual(marshal.load(fp), item)


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


class WarnWhenSubclassedTest(unittest.TestCase):

    def _mywarnings(self, w, category=MyWarning):
        return [x for x in w if x.category is MyWarning]

    def test_no_warning_on_definition(self):
        with warnings.catch_warnings(record=True) as w:
            Deprecated = create_deprecated_class('Deprecated', NewName)

        w = self._mywarnings(w)
        self.assertEqual(w, [])

    def test_subclassing_warning_message(self):
        Deprecated = create_deprecated_class('Deprecated', NewName,
                                             warn_category=MyWarning)

        with warnings.catch_warnings(record=True) as w:
            class UserClass(Deprecated):
                pass

        w = self._mywarnings(w)
        self.assertEqual(len(w), 1)
        self.assertEqual(
            str(w[0].message),
            "tests.test_utils_deprecate.UserClass inherits from "
            "deprecated class tests.test_utils_deprecate.Deprecated, "
            "please inherit from tests.test_utils_deprecate.NewName."
            " (warning only on first subclass, there may be others)"
        )
        self.assertEqual(w[0].lineno, inspect.getsourcelines(UserClass)[1])

    def test_custom_class_paths(self):
        Deprecated = create_deprecated_class('Deprecated', NewName,
                                             new_class_path='foo.NewClass',
                                             old_class_path='bar.OldClass',
                                             warn_category=MyWarning)

        with warnings.catch_warnings(record=True) as w:
            class UserClass(Deprecated):
                pass

            _ = Deprecated()

        w = self._mywarnings(w)
        self.assertEqual(len(w), 2)
        self.assertIn('foo.NewClass', str(w[0].message))
        self.assertIn('bar.OldClass', str(w[0].message))
        self.assertIn('foo.NewClass', str(w[1].message))
        self.assertIn('bar.OldClass', str(w[1].message))

    def test_subclassing_warns_only_on_direct_childs(self):
        Deprecated = create_deprecated_class('Deprecated', NewName,
                                             warn_once=False,
                                             warn_category=MyWarning)

        with warnings.catch_warnings(record=True) as w:
            class UserClass(Deprecated):
                pass

            class NoWarnOnMe(UserClass):
                pass

        w = self._mywarnings(w)
        self.assertEqual(len(w), 1)
        self.assertIn('UserClass', str(w[0].message))

    def test_subclassing_warns_once_by_default(self):
        Deprecated = create_deprecated_class('Deprecated', NewName,
                                             warn_category=MyWarning)

        with warnings.catch_warnings(record=True) as w:
            class UserClass(Deprecated):
                pass

            class FooClass(Deprecated):
                pass

            class BarClass(Deprecated):
                pass

        w = self._mywarnings(w)
        self.assertEqual(len(w), 1)
        self.assertIn('UserClass', str(w[0].message))

    def test_warning_on_instance(self):
        Deprecated = create_deprecated_class('Deprecated', NewName,
                                             warn_category=MyWarning)

        # ignore subclassing warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            class UserClass(Deprecated):
                pass

        with warnings.catch_warnings(record=True) as w:
            _, lineno = Deprecated(), inspect.getlineno(inspect.currentframe())
            _ = UserClass()  # subclass instances don't warn

        w = self._mywarnings(w)
        self.assertEqual(len(w), 1)
        self.assertEqual(
            str(w[0].message),
            "tests.test_utils_deprecate.Deprecated is deprecated, "
            "instantiate tests.test_utils_deprecate.NewName instead."
        )
        self.assertEqual(w[0].lineno, lineno)

    def test_warning_auto_message(self):
        with warnings.catch_warnings(record=True) as w:
            Deprecated = create_deprecated_class('Deprecated', NewName)

            class UserClass2(Deprecated):
                pass

        msg = str(w[0].message)
        self.assertIn("tests.test_utils_deprecate.NewName", msg)
        self.assertIn("tests.test_utils_deprecate.Deprecated", msg)

    def test_issubclass(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            DeprecatedName = create_deprecated_class('DeprecatedName', NewName)

            class UpdatedUserClass1(NewName):
                pass

            class UpdatedUserClass1a(NewName):
                pass

            class OutdatedUserClass1(DeprecatedName):
                pass

            class OutdatedUserClass1a(DeprecatedName):
                pass

            class UnrelatedClass(object):
                pass

            class OldStyleClass:
                pass

        assert issubclass(UpdatedUserClass1, NewName)
        assert issubclass(UpdatedUserClass1a, NewName)
        assert issubclass(UpdatedUserClass1, DeprecatedName)
        assert issubclass(UpdatedUserClass1a, DeprecatedName)
        assert issubclass(OutdatedUserClass1, DeprecatedName)
        assert not issubclass(UnrelatedClass, DeprecatedName)
        assert not issubclass(OldStyleClass, DeprecatedName)
        assert not issubclass(OldStyleClass, DeprecatedName)
        assert not issubclass(OutdatedUserClass1, OutdatedUserClass1a)
        assert not issubclass(OutdatedUserClass1a, OutdatedUserClass1)

        self.assertRaises(TypeError, issubclass, object(), DeprecatedName)

    def test_isinstance(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            DeprecatedName = create_deprecated_class('DeprecatedName', NewName)

            class UpdatedUserClass2(NewName):
                pass

            class UpdatedUserClass2a(NewName):
                pass

            class OutdatedUserClass2(DeprecatedName):
                pass

            class OutdatedUserClass2a(DeprecatedName):
                pass

            class UnrelatedClass(object):
                pass

            class OldStyleClass:
                pass

        assert isinstance(UpdatedUserClass2(), NewName)
        assert isinstance(UpdatedUserClass2a(), NewName)
        assert isinstance(UpdatedUserClass2(), DeprecatedName)
        assert isinstance(UpdatedUserClass2a(), DeprecatedName)
        assert isinstance(OutdatedUserClass2(), DeprecatedName)
        assert isinstance(OutdatedUserClass2a(), DeprecatedName)
        assert not isinstance(OutdatedUserClass2a(), OutdatedUserClass2)
        assert not isinstance(OutdatedUserClass2(), OutdatedUserClass2a)
        assert not isinstance(UnrelatedClass(), DeprecatedName)
        assert not isinstance(OldStyleClass(), DeprecatedName)

    def test_clsdict(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            Deprecated = create_deprecated_class('Deprecated', NewName, {'foo': 'bar'})

        self.assertEqual(Deprecated.foo, 'bar')

    def test_deprecate_a_class_with_custom_metaclass(self):
        Meta1 = type('Meta1', (type,), {})
        New = Meta1('New', (), {})
        Deprecated = create_deprecated_class('Deprecated', New)

    def test_deprecate_subclass_of_deprecated_class(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Deprecated = create_deprecated_class('Deprecated', NewName,
                                                 warn_category=MyWarning)
            AlsoDeprecated = create_deprecated_class('AlsoDeprecated', Deprecated,
                                                     new_class_path='foo.Bar',
                                                     warn_category=MyWarning)

        w = self._mywarnings(w)
        self.assertEqual(len(w), 0, str(map(str, w)))

        with warnings.catch_warnings(record=True) as w:
            AlsoDeprecated()
            class UserClass(AlsoDeprecated):
                pass

        w = self._mywarnings(w)
        self.assertEqual(len(w), 2)
        self.assertIn('AlsoDeprecated', str(w[0].message))
        self.assertIn('foo.Bar', str(w[0].message))
        self.assertIn('AlsoDeprecated', str(w[1].message))
        self.assertIn('foo.Bar', str(w[1].message))

    def test_inspect_stack(self):
        with mock.patch('inspect.stack', side_effect=IndexError):
            with warnings.catch_warnings(record=True) as w:
                DeprecatedName = create_deprecated_class('DeprecatedName', NewName)
                class SubClass(DeprecatedName):
                    pass

        self.assertIn("Error detecting parent module", str(w[0].message))


@mock.patch('scrapy.utils.deprecate.DEPRECATION_RULES',
            [('scrapy.contrib.pipeline.', 'scrapy.pipelines.'),
             ('scrapy.contrib.', 'scrapy.extensions.')])
class ToUnicodeTest(unittest.TestCase):
    def test_converting_an_utf8_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xc3\xb1e'), u'lel\xf1e')

    def test_converting_a_latin_1_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xf1e', 'latin-1'), u'lel\xf1e')

    def test_converting_a_unicode_to_unicode_should_return_the_same_object(self):
        self.assertEqual(to_unicode(u'\xf1e\xf1e\xf1e'), u'\xf1e\xf1e\xf1e')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        self.assertRaises(TypeError, to_unicode, 423)

    def test_errors_argument(self):
        self.assertEqual(
            to_unicode(b'a\xedb', 'utf-8', errors='replace'),
            u'a\ufffdb'
        )


class TestItem(Item):
    name = Field()
    url = Field()
    price = Field()


class SelectorTestCase(unittest.TestCase):

    def test_simple_selection(self):
        """Simple selector tests"""
        body = b"<p><input name='a'value='1'/><input name='b'value='2'/></p>"
        response = TextResponse(url="http://example.com", body=body, encoding='utf-8')
        sel = Selector(response)

        xl = sel.xpath('//input')
        self.assertEqual(2, len(xl))
        for x in xl:
            assert isinstance(x, Selector)

        self.assertEqual(sel.xpath('//input').extract(),
                         [x.extract() for x in sel.xpath('//input')])

        self.assertEqual([x.extract() for x in sel.xpath("//input[@name='a']/@name")],
                         [u'a'])
        self.assertEqual([x.extract() for x in sel.xpath("number(concat(//input[@name='a']/@value, //input[@name='b']/@value))")],
                         [u'12.0'])

        self.assertEqual(sel.xpath("concat('xpath', 'rules')").extract(),
                         [u'xpathrules'])
        self.assertEqual([x.extract() for x in sel.xpath("concat(//input[@name='a']/@value, //input[@name='b']/@value)")],
                         [u'12'])

    def test_root_base_url(self):
        body = b'<html><form action="/path"><input name="a" /></form></html>'
        url = "http://example.com"
        response = TextResponse(url=url, body=body, encoding='utf-8')
        sel = Selector(response)
        self.assertEqual(url, sel.root.base)

    def test_deprecated_root_argument(self):
        with warnings.catch_warnings(record=True) as w:
            root = etree.fromstring(u'<html/>')
            sel = Selector(_root=root)
            self.assertIs(root, sel.root)
            self.assertEqual(str(w[-1].message),
                             'Argument `_root` is deprecated, use `root` instead')

    def test_deprecated_root_argument_ambiguous(self):
        with warnings.catch_warnings(record=True) as w:
            _root = etree.fromstring(u'<xml/>')
            root = etree.fromstring(u'<html/>')
            sel = Selector(_root=_root, root=root)
            self.assertIs(root, sel.root)
            self.assertIn('Ignoring deprecated `_root` argument', str(w[-1].message))

    def test_flavor_detection(self):
        text = b'<div><img src="a.jpg"><p>Hello</div>'
        sel = Selector(XmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'xml')
        self.assertEqual(sel.xpath("//div").extract(),
                         [u'<div><img src="a.jpg"><p>Hello</p></img></div>'])

        sel = Selector(HtmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'html')
        self.assertEqual(sel.xpath("//div").extract(),
                         [u'<div><img src="a.jpg"><p>Hello</p></div>'])

    def test_http_header_encoding_precedence(self):
        # u'\xa3'     = pound symbol in unicode
        # u'\xc2\xa3' = pound symbol in utf-8
        # u'\xa3'     = pound symbol in latin-1 (iso-8859-1)

        meta = u'<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">'
        head = u'<head>' + meta + u'</head>'
        body_content = u'<span id="blank">\xa3</span>'
        body = u'<body>' + body_content + u'</body>'
        html = u'<html>' + head + body + u'</html>'
        encoding = 'utf-8'
        html_utf8 = html.encode(encoding)

        headers = {'Content-Type': ['text/html; charset=utf-8']}
        response = HtmlResponse(url="http://example.com", headers=headers, body=html_utf8)
        x = Selector(response)
        self.assertEqual(x.xpath("//span[@id='blank']/text()").extract(),
                          [u'\xa3'])

    def test_badly_encoded_body(self):
        # \xe9 alone isn't valid utf8 sequence
        r1 = TextResponse('http://www.example.com', \
                          body=b'<html><p>an Jos\xe9 de</p><html>', \
                          encoding='utf-8')
        Selector(r1).xpath('//text()').extract()

    def test_weakref_slots(self):
        """Check that classes are using slots and are weak-referenceable"""
        x = Selector(text='')
        weakref.ref(x)
        assert not hasattr(x, '__dict__'), "%s does not use __slots__" % \
            x.__class__.__name__

    def test_deprecated_selector_methods(self):
        sel = Selector(TextResponse(url="http://example.com", body=b'<p>some text</p>'))

        with warnings.catch_warnings(record=True) as w:
            sel.select('//p')
            self.assertSubstring('Use .xpath() instead', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            sel.extract_unquoted()
            self.assertSubstring('Use .extract() instead', str(w[-1].message))

    def test_deprecated_selectorlist_methods(self):
        sel = Selector(TextResponse(url="http://example.com", body=b'<p>some text</p>'))

        with warnings.catch_warnings(record=True) as w:
            sel.xpath('//p').select('.')
            self.assertSubstring('Use .xpath() instead', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            sel.xpath('//p').extract_unquoted()
            self.assertSubstring('Use .extract() instead', str(w[-1].message))

    def test_selector_bad_args(self):
        with self.assertRaisesRegexp(ValueError, 'received both response and text'):
            Selector(TextResponse(url='http://example.com', body=b''), text=u'')


class TestRequestMetaSettingFallback(TestCase):

    params = [
        (
            # When an unknown policy is referenced in Request.meta
            # (here, a typo error),
            # the policy defined in settings takes precedence
            {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'},
            {},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # same as above but with string value for settings policy
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # request meta references a wrong policy but it is set,
            # so the Referrer-Policy header in response is not used,
            # and the settings' policy is applied
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unsafe-url'},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # here, request meta does not set the policy
            # so response headers take precedence
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unsafe-url'},
            {},
            UnsafeUrlPolicy,
            False
        ),
        (
            # here, request meta does not set the policy,
            # but response headers also use an unknown policy,
            # so the settings' policy is used
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unknown'},
            {},
            OriginWhenCrossOriginPolicy,
            True
        )
    ]

    def test(self):

        origin = 'http://www.scrapy.org'
        target = 'http://www.example.com'

        for settings, response_headers, request_meta, policy_class, check_warning in self.params[3:]:
            spider = Spider('foo')
            mw = RefererMiddleware(Settings(settings))

            response = Response(origin, headers=response_headers)
            request = Request(target, meta=request_meta)

            with warnings.catch_warnings(record=True) as w:
                policy = mw.policy(response, request)
                self.assertIsInstance(policy, policy_class)

                if check_warning:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(w[0].category, RuntimeWarning, w[0].message)


class MediaPipelineTestCase(BaseMediaPipelineTestCase):

    pipeline_class = MockedMediaPipeline

    @inlineCallbacks
    def test_result_succeed(self):
        cb = lambda _: self.pipe._mockcalled.append('request_callback') or _
        eb = lambda _: self.pipe._mockcalled.append('request_errback') or _
        rsp = Response('http://url1')
        req = Request('http://url1', meta=dict(response=rsp), callback=cb, errback=eb)
        item = dict(requests=req)
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertEqual(new_item['results'], [(True, rsp)])
        self.assertEqual(self.pipe._mockcalled,
                ['get_media_requests', 'media_to_download',
                    'media_downloaded', 'request_callback', 'item_completed'])

    @inlineCallbacks
    def test_result_failure(self):
        self.pipe.LOG_FAILED_RESULTS = False
        cb = lambda _: self.pipe._mockcalled.append('request_callback') or _
        eb = lambda _: self.pipe._mockcalled.append('request_errback') or _
        fail = Failure(Exception())
        req = Request('http://url1', meta=dict(response=fail), callback=cb, errback=eb)
        item = dict(requests=req)
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertEqual(new_item['results'], [(False, fail)])
        self.assertEqual(self.pipe._mockcalled,
                ['get_media_requests', 'media_to_download',
                    'media_failed', 'request_errback', 'item_completed'])

    @inlineCallbacks
    def test_mix_of_success_and_failure(self):
        self.pipe.LOG_FAILED_RESULTS = False
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        fail = Failure(Exception())
        req2 = Request('http://url2', meta=dict(response=fail))
        item = dict(requests=[req1, req2])
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertEqual(new_item['results'], [(True, rsp1), (False, fail)])
        m = self.pipe._mockcalled
        # only once
        self.assertEqual(m[0], 'get_media_requests') # first hook called
        self.assertEqual(m.count('get_media_requests'), 1)
        self.assertEqual(m.count('item_completed'), 1)
        self.assertEqual(m[-1], 'item_completed') # last hook called
        # twice, one per request
        self.assertEqual(m.count('media_to_download'), 2)
        # one to handle success and other for failure
        self.assertEqual(m.count('media_downloaded'), 1)
        self.assertEqual(m.count('media_failed'), 1)

    @inlineCallbacks
    def test_get_media_requests(self):
        # returns single Request (without callback)
        req = Request('http://url')
        item = dict(requests=req) # pass a single item
        new_item = yield self.pipe.process_item(item, self.spider)
        assert new_item is item
        assert request_fingerprint(req) in self.info.downloaded

        # returns iterable of Requests
        req1 = Request('http://url1')
        req2 = Request('http://url2')
        item = dict(requests=iter([req1, req2]))
        new_item = yield self.pipe.process_item(item, self.spider)
        assert new_item is item
        assert request_fingerprint(req1) in self.info.downloaded
        assert request_fingerprint(req2) in self.info.downloaded

    @inlineCallbacks
    def test_results_are_cached_across_multiple_items(self):
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        item = dict(requests=req1)
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertTrue(new_item is item)
        self.assertEqual(new_item['results'], [(True, rsp1)])

        # rsp2 is ignored, rsp1 must be in results because request fingerprints are the same
        req2 = Request(req1.url, meta=dict(response=Response('http://donot.download.me')))
        item = dict(requests=req2)
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertTrue(new_item is item)
        self.assertEqual(request_fingerprint(req1), request_fingerprint(req2))
        self.assertEqual(new_item['results'], [(True, rsp1)])

    @inlineCallbacks
    def test_results_are_cached_for_requests_of_single_item(self):
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        req2 = Request(req1.url, meta=dict(response=Response('http://donot.download.me')))
        item = dict(requests=[req1, req2])
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertTrue(new_item is item)
        self.assertEqual(new_item['results'], [(True, rsp1), (True, rsp1)])

    @inlineCallbacks
    def test_wait_if_request_is_downloading(self):
        def _check_downloading(response):
            fp = request_fingerprint(req1)
            self.assertTrue(fp in self.info.downloading)
            self.assertTrue(fp in self.info.waiting)
            self.assertTrue(fp not in self.info.downloaded)
            self.assertEqual(len(self.info.waiting[fp]), 2)
            return response

        rsp1 = Response('http://url')
        def rsp1_func():
            dfd = Deferred().addCallback(_check_downloading)
            reactor.callLater(.1, dfd.callback, rsp1)
            return dfd

        def rsp2_func():
            self.fail('it must cache rsp1 result and must not try to redownload')

        req1 = Request('http://url', meta=dict(response=rsp1_func))
        req2 = Request(req1.url, meta=dict(response=rsp2_func))
        item = dict(requests=[req1, req2])
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertEqual(new_item['results'], [(True, rsp1), (True, rsp1)])

    @inlineCallbacks
    def test_use_media_to_download_result(self):
        req = Request('http://url', meta=dict(result='ITSME', response=self.fail))
        item = dict(requests=req)
        new_item = yield self.pipe.process_item(item, self.spider)
        self.assertEqual(new_item['results'], [(True, 'ITSME')])
        self.assertEqual(self.pipe._mockcalled, \
                ['get_media_requests', 'media_to_download', 'item_completed'])

