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


class Https11WrongHostnameTestCase(Http11TestCase):
    scheme = 'https'

    # above tests use a server certificate for "localhost",
    # client connection to "localhost" too.
    # here we test that even if the server certificate is for another domain,
    # "www.example.com" in this case,
    # the tests still pass
    keyfile = 'keys/example-com.key.pem'
    certfile = 'keys/example-com.cert.pem'


class CustomPythonOrgPolicy(ReferrerPolicy):
    """
    A dummy policy that returns referrer as http(s)://python.org
    depending on the scheme of the target URL.
    """
    def referrer(self, response, request):
        scheme = urlparse(request).scheme
        if scheme == 'https':
            return b'https://python.org/'
        elif scheme == 'http':
            return b'http://python.org/'


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

class MarshalLifoDiskQueueTest(t.LifoDiskQueueTest):

    def queue(self):
        return MarshalLifoDiskQueue(self.qpath)

    def test_serialize(self):
        q = self.queue()
        q.push('a')
        q.push(123)
        q.push({'a': 'dict'})
        self.assertEqual(q.pop(), {'a': 'dict'})
        self.assertEqual(q.pop(), 123)
        self.assertEqual(q.pop(), 'a')

    test_nonserializable_object = nonserializable_object_test


class FormRequestTest(RequestTest):

    request_class = FormRequest

    def assertQueryEqual(self, first, second, msg=None):
        first = to_native_str(first).split("&")
        second = to_native_str(second).split("&")
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_empty_formdata(self):
        r1 = self.request_class("http://www.example.com", formdata={})
        self.assertEqual(r1.body, b'')

    def test_default_encoding_bytes(self):
        # using default encoding (utf-8)
        data = {b'one': b'two', b'price': b'\xc2\xa3 100'}
        r2 = self.request_class("http://www.example.com", formdata=data)
        self.assertEqual(r2.method, 'POST')
        self.assertEqual(r2.encoding, 'utf-8')
        self.assertQueryEqual(r2.body, b'price=%C2%A3+100&one=two')
        self.assertEqual(r2.headers[b'Content-Type'], b'application/x-www-form-urlencoded')

    def test_default_encoding_textual_data(self):
        # using default encoding (utf-8)
        data = {u'µ one': u'two', u'price': u'£ 100'}
        r2 = self.request_class("http://www.example.com", formdata=data)
        self.assertEqual(r2.method, 'POST')
        self.assertEqual(r2.encoding, 'utf-8')
        self.assertQueryEqual(r2.body, b'price=%C2%A3+100&%C2%B5+one=two')
        self.assertEqual(r2.headers[b'Content-Type'], b'application/x-www-form-urlencoded')

    def test_default_encoding_mixed_data(self):
        # using default encoding (utf-8)
        data = {u'\u00b5one': b'two', b'price\xc2\xa3': u'\u00a3 100'}
        r2 = self.request_class("http://www.example.com", formdata=data)
        self.assertEqual(r2.method, 'POST')
        self.assertEqual(r2.encoding, 'utf-8')
        self.assertQueryEqual(r2.body, b'%C2%B5one=two&price%C2%A3=%C2%A3+100')
        self.assertEqual(r2.headers[b'Content-Type'], b'application/x-www-form-urlencoded')

    def test_custom_encoding_bytes(self):
        data = {b'\xb5 one': b'two', b'price': b'\xa3 100'}
        r2 = self.request_class("http://www.example.com", formdata=data,
                                    encoding='latin1')
        self.assertEqual(r2.method, 'POST')
        self.assertEqual(r2.encoding, 'latin1')
        self.assertQueryEqual(r2.body, b'price=%A3+100&%B5+one=two')
        self.assertEqual(r2.headers[b'Content-Type'], b'application/x-www-form-urlencoded')

    def test_custom_encoding_textual_data(self):
        data = {'price': u'£ 100'}
        r3 = self.request_class("http://www.example.com", formdata=data,
                                    encoding='latin1')
        self.assertEqual(r3.encoding, 'latin1')
        self.assertEqual(r3.body, b'price=%A3+100')

    def test_multi_key_values(self):
        # using multiples values for a single key
        data = {'price': u'\xa3 100', 'colours': ['red', 'blue', 'green']}
        r3 = self.request_class("http://www.example.com", formdata=data)
        self.assertQueryEqual(r3.body,
            b'colours=red&colours=blue&colours=green&price=%C2%A3+100')

    def test_from_response_post(self):
        response = _buildresponse(
            b"""<form action="post.php" method="POST">
            <input type="hidden" name="test" value="val1">
            <input type="hidden" name="test" value="val2">
            <input type="hidden" name="test2" value="xxx">
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'})

        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers[b'Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req)
        self.assertEqual(set(fs[b'test']), {b'val1', b'val2'})
        self.assertEqual(set(fs[b'one']), {b'two', b'three'})
        self.assertEqual(fs[b'test2'], [b'xxx'])
        self.assertEqual(fs[b'six'], [b'seven'])

    def test_from_response_post_nonascii_bytes_utf8(self):
        response = _buildresponse(
            b"""<form action="post.php" method="POST">
            <input type="hidden" name="test \xc2\xa3" value="val1">
            <input type="hidden" name="test \xc2\xa3" value="val2">
            <input type="hidden" name="test2" value="xxx \xc2\xb5">
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'})

        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers[b'Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req, to_unicode=True)
        self.assertEqual(set(fs[u'test £']), {u'val1', u'val2'})
        self.assertEqual(set(fs[u'one']), {u'two', u'three'})
        self.assertEqual(fs[u'test2'], [u'xxx µ'])
        self.assertEqual(fs[u'six'], [u'seven'])

    def test_from_response_post_nonascii_bytes_latin1(self):
        response = _buildresponse(
            b"""<form action="post.php" method="POST">
            <input type="hidden" name="test \xa3" value="val1">
            <input type="hidden" name="test \xa3" value="val2">
            <input type="hidden" name="test2" value="xxx \xb5">
            </form>""",
            url="http://www.example.com/this/list.html",
            encoding='latin1',
            )
        req = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'})

        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers[b'Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req, to_unicode=True, encoding='latin1')
        self.assertEqual(set(fs[u'test £']), {u'val1', u'val2'})
        self.assertEqual(set(fs[u'one']), {u'two', u'three'})
        self.assertEqual(fs[u'test2'], [u'xxx µ'])
        self.assertEqual(fs[u'six'], [u'seven'])

    def test_from_response_post_nonascii_unicode(self):
        response = _buildresponse(
            u"""<form action="post.php" method="POST">
            <input type="hidden" name="test £" value="val1">
            <input type="hidden" name="test £" value="val2">
            <input type="hidden" name="test2" value="xxx µ">
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'})

        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers[b'Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req, to_unicode=True)
        self.assertEqual(set(fs[u'test £']), {u'val1', u'val2'})
        self.assertEqual(set(fs[u'one']), {u'two', u'three'})
        self.assertEqual(fs[u'test2'], [u'xxx µ'])
        self.assertEqual(fs[u'six'], [u'seven'])

    def test_from_response_extra_headers(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="test" value="val1">
            <input type="hidden" name="test" value="val2">
            <input type="hidden" name="test2" value="xxx">
            </form>""")
        req = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'},
                headers={"Accept-Encoding": "gzip,deflate"})
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers['Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.headers['Accept-Encoding'], b'gzip,deflate')

    def test_from_response_get(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="hidden" name="test" value="val1">
            <input type="hidden" name="test" value="val2">
            <input type="hidden" name="test2" value="xxx">
            </form>""",
            url="http://www.example.com/this/list.html")
        r1 = self.request_class.from_response(response,
                formdata={'one': ['two', 'three'], 'six': 'seven'})
        self.assertEqual(r1.method, 'GET')
        self.assertEqual(urlparse(r1.url).hostname, "www.example.com")
        self.assertEqual(urlparse(r1.url).path, "/this/get.php")
        fs = _qs(r1)
        self.assertEqual(set(fs[b'test']), set([b'val1', b'val2']))
        self.assertEqual(set(fs[b'one']), set([b'two', b'three']))
        self.assertEqual(fs[b'test2'], [b'xxx'])
        self.assertEqual(fs[b'six'], [b'seven'])

    def test_from_response_override_params(self):
        response = _buildresponse(
            """<form action="get.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            </form>""")
        req = self.request_class.from_response(response, formdata={'two': '2'})
        fs = _qs(req)
        self.assertEqual(fs[b'one'], [b'1'])
        self.assertEqual(fs[b'two'], [b'2'])

    def test_from_response_drop_params(self):
        response = _buildresponse(
            """<form action="get.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            </form>""")
        req = self.request_class.from_response(response, formdata={'two': None})
        fs = _qs(req)
        self.assertEqual(fs[b'one'], [b'1'])
        self.assertNotIn(b'two', fs)

    def test_from_response_override_method(self):
        response = _buildresponse(
                '''<html><body>
                <form action="/app"></form>
                </body></html>''')
        request = FormRequest.from_response(response)
        self.assertEqual(request.method, 'GET')
        request = FormRequest.from_response(response, method='POST')
        self.assertEqual(request.method, 'POST')

    def test_from_response_override_url(self):
        response = _buildresponse(
                '''<html><body>
                <form action="/app"></form>
                </body></html>''')
        request = FormRequest.from_response(response)
        self.assertEqual(request.url, 'http://example.com/app')
        request = FormRequest.from_response(response, url='http://foo.bar/absolute')
        self.assertEqual(request.url, 'http://foo.bar/absolute')
        request = FormRequest.from_response(response, url='/relative')
        self.assertEqual(request.url, 'http://example.com/relative')

    def test_from_response_case_insensitive(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="SuBmIt" name="clickable1" value="clicked1">
            <input type="iMaGe" name="i1" src="http://my.image.org/1.jpg">
            <input type="submit" name="clickable2" value="clicked2">
            </form>""")
        req = self.request_class.from_response(response)
        fs = _qs(req)
        self.assertEqual(fs[b'clickable1'], [b'clicked1'])
        self.assertFalse(b'i1' in fs, fs)  # xpath in _get_inputs()
        self.assertFalse(b'clickable2' in fs, fs)  # xpath in _get_clickable()

    def test_from_response_submit_first_clickable(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="submit" name="clickable1" value="clicked1">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            <input type="submit" name="clickable2" value="clicked2">
            </form>""")
        req = self.request_class.from_response(response, formdata={'two': '2'})
        fs = _qs(req)
        self.assertEqual(fs[b'clickable1'], [b'clicked1'])
        self.assertFalse(b'clickable2' in fs, fs)
        self.assertEqual(fs[b'one'], [b'1'])
        self.assertEqual(fs[b'two'], [b'2'])

    def test_from_response_submit_not_first_clickable(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="submit" name="clickable1" value="clicked1">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            <input type="submit" name="clickable2" value="clicked2">
            </form>""")
        req = self.request_class.from_response(response, formdata={'two': '2'}, \
                                              clickdata={'name': 'clickable2'})
        fs = _qs(req)
        self.assertEqual(fs[b'clickable2'], [b'clicked2'])
        self.assertFalse(b'clickable1' in fs, fs)
        self.assertEqual(fs[b'one'], [b'1'])
        self.assertEqual(fs[b'two'], [b'2'])

    def test_from_response_dont_submit_image_as_input(self):
        response = _buildresponse(
            """<form>
            <input type="hidden" name="i1" value="i1v">
            <input type="image" name="i2" src="http://my.image.org/1.jpg">
            <input type="submit" name="i3" value="i3v">
            </form>""")
        req = self.request_class.from_response(response, dont_click=True)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'i1v']})

    def test_from_response_dont_submit_reset_as_input(self):
        response = _buildresponse(
            """<form>
            <input type="hidden" name="i1" value="i1v">
            <input type="text" name="i2" value="i2v">
            <input type="reset" name="resetme">
            <input type="submit" name="i3" value="i3v">
            </form>""")
        req = self.request_class.from_response(response, dont_click=True)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'i1v'], b'i2': [b'i2v']})

    def test_from_response_multiple_clickdata(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="submit" name="clickable" value="clicked1">
            <input type="submit" name="clickable" value="clicked2">
            <input type="hidden" name="one" value="clicked1">
            <input type="hidden" name="two" value="clicked2">
            </form>""")
        req = self.request_class.from_response(response, \
                clickdata={u'name': u'clickable', u'value': u'clicked2'})
        fs = _qs(req)
        self.assertEqual(fs[b'clickable'], [b'clicked2'])
        self.assertEqual(fs[b'one'], [b'clicked1'])
        self.assertEqual(fs[b'two'], [b'clicked2'])

    def test_from_response_unicode_clickdata(self):
        response = _buildresponse(
            u"""<form action="get.php" method="GET">
            <input type="submit" name="price in \u00a3" value="\u00a3 1000">
            <input type="submit" name="price in \u20ac" value="\u20ac 2000">
            <input type="hidden" name="poundsign" value="\u00a3">
            <input type="hidden" name="eurosign" value="\u20ac">
            </form>""")
        req = self.request_class.from_response(response, \
                clickdata={u'name': u'price in \u00a3'})
        fs = _qs(req, to_unicode=True)
        self.assertTrue(fs[u'price in \u00a3'])

    def test_from_response_unicode_clickdata_latin1(self):
        response = _buildresponse(
            u"""<form action="get.php" method="GET">
            <input type="submit" name="price in \u00a3" value="\u00a3 1000">
            <input type="submit" name="price in \u00a5" value="\u00a5 2000">
            <input type="hidden" name="poundsign" value="\u00a3">
            <input type="hidden" name="yensign" value="\u00a5">
            </form>""",
            encoding='latin1')
        req = self.request_class.from_response(response, \
                clickdata={u'name': u'price in \u00a5'})
        fs = _qs(req, to_unicode=True, encoding='latin1')
        self.assertTrue(fs[u'price in \u00a5'])

    def test_from_response_multiple_forms_clickdata(self):
        response = _buildresponse(
            """<form name="form1">
            <input type="submit" name="clickable" value="clicked1">
            <input type="hidden" name="field1" value="value1">
            </form>
            <form name="form2">
            <input type="submit" name="clickable" value="clicked2">
            <input type="hidden" name="field2" value="value2">
            </form>
            """)
        req = self.request_class.from_response(response, formname='form2', \
                clickdata={u'name': u'clickable'})
        fs = _qs(req)
        self.assertEqual(fs[b'clickable'], [b'clicked2'])
        self.assertEqual(fs[b'field2'], [b'value2'])
        self.assertFalse(b'field1' in fs, fs)

    def test_from_response_override_clickable(self):
        response = _buildresponse('''<form><input type="submit" name="clickme" value="one"> </form>''')
        req = self.request_class.from_response(response, \
                formdata={'clickme': 'two'}, clickdata={'name': 'clickme'})
        fs = _qs(req)
        self.assertEqual(fs[b'clickme'], [b'two'])

    def test_from_response_dont_click(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="submit" name="clickable1" value="clicked1">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            <input type="submit" name="clickable2" value="clicked2">
            </form>""")
        r1 = self.request_class.from_response(response, dont_click=True)
        fs = _qs(r1)
        self.assertFalse(b'clickable1' in fs, fs)
        self.assertFalse(b'clickable2' in fs, fs)

    def test_from_response_ambiguous_clickdata(self):
        response = _buildresponse(
            """
            <form action="get.php" method="GET">
            <input type="submit" name="clickable1" value="clicked1">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="3">
            <input type="submit" name="clickable2" value="clicked2">
            </form>""")
        self.assertRaises(ValueError, self.request_class.from_response,
                          response, clickdata={'type': 'submit'})

    def test_from_response_non_matching_clickdata(self):
        response = _buildresponse(
            """<form>
            <input type="submit" name="clickable" value="clicked">
            </form>""")
        self.assertRaises(ValueError, self.request_class.from_response,
                          response, clickdata={'nonexistent': 'notme'})

    def test_from_response_nr_index_clickdata(self):
        response = _buildresponse(
            """<form>
            <input type="submit" name="clickable1" value="clicked1">
            <input type="submit" name="clickable2" value="clicked2">
            </form>
            """)
        req = self.request_class.from_response(response, clickdata={'nr': 1})
        fs = _qs(req)
        self.assertIn(b'clickable2', fs)
        self.assertNotIn(b'clickable1', fs)

    def test_from_response_invalid_nr_index_clickdata(self):
        response = _buildresponse(
            """<form>
            <input type="submit" name="clickable" value="clicked">
            </form>
            """)
        self.assertRaises(ValueError, self.request_class.from_response,
                          response, clickdata={'nr': 1})

    def test_from_response_errors_noform(self):
        response = _buildresponse("""<html></html>""")
        self.assertRaises(ValueError, self.request_class.from_response, response)

    def test_from_response_invalid_html5(self):
        response = _buildresponse("""<!DOCTYPE html><body></html><form>"""
                                  """<input type="text" name="foo" value="xxx">"""
                                  """</form></body></html>""")
        req = self.request_class.from_response(response, formdata={'bar': 'buz'})
        fs = _qs(req)
        self.assertEqual(fs, {b'foo': [b'xxx'], b'bar': [b'buz']})

    def test_from_response_errors_formnumber(self):
        response = _buildresponse(
            """<form action="get.php" method="GET">
            <input type="hidden" name="test" value="val1">
            <input type="hidden" name="test" value="val2">
            <input type="hidden" name="test2" value="xxx">
            </form>""")
        self.assertRaises(IndexError, self.request_class.from_response, response, formnumber=1)

    def test_from_response_noformname(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>""")
        r1 = self.request_class.from_response(response, formdata={'two':'3'})
        self.assertEqual(r1.method, 'POST')
        self.assertEqual(r1.headers['Content-type'], b'application/x-www-form-urlencoded')
        fs = _qs(r1)
        self.assertEqual(fs, {b'one': [b'1'], b'two': [b'3']})

    def test_from_response_formname_exists(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>
            <form name="form2" action="post.php" method="POST">
            <input type="hidden" name="three" value="3">
            <input type="hidden" name="four" value="4">
            </form>""")
        r1 = self.request_class.from_response(response, formname="form2")
        self.assertEqual(r1.method, 'POST')
        fs = _qs(r1)
        self.assertEqual(fs, {b'four': [b'4'], b'three': [b'3']})

    def test_from_response_formname_notexist(self):
        response = _buildresponse(
            """<form name="form1" action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            </form>
            <form name="form2" action="post.php" method="POST">
            <input type="hidden" name="two" value="2">
            </form>""")
        r1 = self.request_class.from_response(response, formname="form3")
        self.assertEqual(r1.method, 'POST')
        fs = _qs(r1)
        self.assertEqual(fs, {b'one': [b'1']})

    def test_from_response_formname_errors_formnumber(self):
        response = _buildresponse(
            """<form name="form1" action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            </form>
            <form name="form2" action="post.php" method="POST">
            <input type="hidden" name="two" value="2">
            </form>""")
        self.assertRaises(IndexError, self.request_class.from_response, \
                          response, formname="form3", formnumber=2)

    def test_from_response_formid_exists(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>
            <form id="form2" action="post.php" method="POST">
            <input type="hidden" name="three" value="3">
            <input type="hidden" name="four" value="4">
            </form>""")
        r1 = self.request_class.from_response(response, formid="form2")
        self.assertEqual(r1.method, 'POST')
        fs = _qs(r1)
        self.assertEqual(fs, {b'four': [b'4'], b'three': [b'3']})

    def test_from_response_formname_notexists_fallback_formid(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>
            <form id="form2" name="form2" action="post.php" method="POST">
            <input type="hidden" name="three" value="3">
            <input type="hidden" name="four" value="4">
            </form>""")
        r1 = self.request_class.from_response(response, formname="form3", formid="form2")
        self.assertEqual(r1.method, 'POST')
        fs = _qs(r1)
        self.assertEqual(fs, {b'four': [b'4'], b'three': [b'3']})

    def test_from_response_formid_notexist(self):
        response = _buildresponse(
            """<form id="form1" action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            </form>
            <form id="form2" action="post.php" method="POST">
            <input type="hidden" name="two" value="2">
            </form>""")
        r1 = self.request_class.from_response(response, formid="form3")
        self.assertEqual(r1.method, 'POST')
        fs = _qs(r1)
        self.assertEqual(fs, {b'one': [b'1']})

    def test_from_response_formid_errors_formnumber(self):
        response = _buildresponse(
            """<form id="form1" action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            </form>
            <form id="form2" name="form2" action="post.php" method="POST">
            <input type="hidden" name="two" value="2">
            </form>""")
        self.assertRaises(IndexError, self.request_class.from_response, \
                          response, formid="form3", formnumber=2)

    def test_from_response_select(self):
        res = _buildresponse(
            '''<form>
            <select name="i1">
                <option value="i1v1">option 1</option>
                <option value="i1v2" selected>option 2</option>
            </select>
            <select name="i2">
                <option value="i2v1">option 1</option>
                <option value="i2v2">option 2</option>
            </select>
            <select>
                <option value="i3v1">option 1</option>
                <option value="i3v2">option 2</option>
            </select>
            <select name="i4" multiple>
                <option value="i4v1">option 1</option>
                <option value="i4v2" selected>option 2</option>
                <option value="i4v3" selected>option 3</option>
            </select>
            <select name="i5" multiple>
                <option value="i5v1">option 1</option>
                <option value="i5v2">option 2</option>
            </select>
            <select name="i6"></select>
            <select name="i7"/>
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req, to_unicode=True)
        self.assertEqual(fs, {'i1': ['i1v2'], 'i2': ['i2v1'], 'i4': ['i4v2', 'i4v3']})

    def test_from_response_radio(self):
        res = _buildresponse(
            '''<form>
            <input type="radio" name="i1" value="i1v1">
            <input type="radio" name="i1" value="iv2" checked>
            <input type="radio" name="i2" checked>
            <input type="radio" name="i2">
            <input type="radio" name="i3" value="i3v1">
            <input type="radio" name="i3">
            <input type="radio" value="i4v1">
            <input type="radio">
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'iv2'], b'i2': [b'on']})

    def test_from_response_checkbox(self):
        res = _buildresponse(
            '''<form>
            <input type="checkbox" name="i1" value="i1v1">
            <input type="checkbox" name="i1" value="iv2" checked>
            <input type="checkbox" name="i2" checked>
            <input type="checkbox" name="i2">
            <input type="checkbox" name="i3" value="i3v1">
            <input type="checkbox" name="i3">
            <input type="checkbox" value="i4v1">
            <input type="checkbox">
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'iv2'], b'i2': [b'on']})

    def test_from_response_input_text(self):
        res = _buildresponse(
            '''<form>
            <input type="text" name="i1" value="i1v1">
            <input type="text" name="i2">
            <input type="text" value="i3v1">
            <input type="text">
            <input name="i4" value="i4v1">
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'i1v1'], b'i2': [b''], b'i4': [b'i4v1']})

    def test_from_response_input_hidden(self):
        res = _buildresponse(
            '''<form>
            <input type="hidden" name="i1" value="i1v1">
            <input type="hidden" name="i2">
            <input type="hidden" value="i3v1">
            <input type="hidden">
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'i1v1'], b'i2': [b'']})

    def test_from_response_input_textarea(self):
        res = _buildresponse(
            '''<form>
            <textarea name="i1">i1v</textarea>
            <textarea name="i2"></textarea>
            <textarea name="i3"/>
            <textarea>i4v</textarea>
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(fs, {b'i1': [b'i1v'], b'i2': [b''], b'i3': [b'']})

    def test_from_response_descendants(self):
        res = _buildresponse(
            '''<form>
            <div>
              <fieldset>
                <input type="text" name="i1">
                <select name="i2">
                    <option value="v1" selected>
                </select>
              </fieldset>
              <input type="radio" name="i3" value="i3v2" checked>
              <input type="checkbox" name="i4" value="i4v2" checked>
              <textarea name="i5"></textarea>
              <input type="hidden" name="h1" value="h1v">
              </div>
            <input type="hidden" name="h2" value="h2v">
            </form>''')
        req = self.request_class.from_response(res)
        fs = _qs(req)
        self.assertEqual(set(fs), set([b'h2', b'i2', b'i1', b'i3', b'h1', b'i5', b'i4']))

    def test_from_response_xpath(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>
            <form action="post2.php" method="POST">
            <input type="hidden" name="three" value="3">
            <input type="hidden" name="four" value="4">
            </form>""")
        r1 = self.request_class.from_response(response, formxpath="//form[@action='post.php']")
        fs = _qs(r1)
        self.assertEqual(fs[b'one'], [b'1'])

        r1 = self.request_class.from_response(response, formxpath="//form/input[@name='four']")
        fs = _qs(r1)
        self.assertEqual(fs[b'three'], [b'3'])

        self.assertRaises(ValueError, self.request_class.from_response,
                          response, formxpath="//form/input[@name='abc']")

    def test_from_response_unicode_xpath(self):
        response = _buildresponse(b'<form name="\xd1\x8a"></form>')
        r = self.request_class.from_response(response, formxpath=u"//form[@name='\u044a']")
        fs = _qs(r)
        self.assertEqual(fs, {})

        xpath = u"//form[@name='\u03b1']"
        encoded = xpath if six.PY3 else xpath.encode('unicode_escape')
        self.assertRaisesRegexp(ValueError, re.escape(encoded),
                                self.request_class.from_response,
                                response, formxpath=xpath)

    def test_from_response_button_submit(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="test1" value="val1">
            <input type="hidden" name="test2" value="val2">
            <button type="submit" name="button1" value="submit1">Submit</button>
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response)
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers['Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req)
        self.assertEqual(fs[b'test1'], [b'val1'])
        self.assertEqual(fs[b'test2'], [b'val2'])
        self.assertEqual(fs[b'button1'], [b'submit1'])

    def test_from_response_button_notype(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="test1" value="val1">
            <input type="hidden" name="test2" value="val2">
            <button name="button1" value="submit1">Submit</button>
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response)
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers['Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req)
        self.assertEqual(fs[b'test1'], [b'val1'])
        self.assertEqual(fs[b'test2'], [b'val2'])
        self.assertEqual(fs[b'button1'], [b'submit1'])

    def test_from_response_submit_novalue(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="test1" value="val1">
            <input type="hidden" name="test2" value="val2">
            <input type="submit" name="button1">Submit</button>
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response)
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers['Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req)
        self.assertEqual(fs[b'test1'], [b'val1'])
        self.assertEqual(fs[b'test2'], [b'val2'])
        self.assertEqual(fs[b'button1'], [b''])

    def test_from_response_button_novalue(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="test1" value="val1">
            <input type="hidden" name="test2" value="val2">
            <button type="submit" name="button1">Submit</button>
            </form>""",
            url="http://www.example.com/this/list.html")
        req = self.request_class.from_response(response)
        self.assertEqual(req.method, 'POST')
        self.assertEqual(req.headers['Content-type'], b'application/x-www-form-urlencoded')
        self.assertEqual(req.url, "http://www.example.com/this/post.php")
        fs = _qs(req)
        self.assertEqual(fs[b'test1'], [b'val1'])
        self.assertEqual(fs[b'test2'], [b'val2'])
        self.assertEqual(fs[b'button1'], [b''])

    def test_html_base_form_action(self):
        response = _buildresponse(
            """
            <html>
                <head>
                    <base href=" http://b.com/">
                </head>
                <body>
                    <form action="test_form">
                    </form>
                </body>
            </html>
            """,
            url='http://a.com/'
        )
        req = self.request_class.from_response(response)
        self.assertEqual(req.url, 'http://b.com/test_form')

    def test_spaces_in_action(self):
        resp = _buildresponse('<body><form action=" path\n"></form></body>')
        req = self.request_class.from_response(resp)
        self.assertEqual(req.url, 'http://example.com/path')

    def test_from_response_css(self):
        response = _buildresponse(
            """<form action="post.php" method="POST">
            <input type="hidden" name="one" value="1">
            <input type="hidden" name="two" value="2">
            </form>
            <form action="post2.php" method="POST">
            <input type="hidden" name="three" value="3">
            <input type="hidden" name="four" value="4">
            </form>""")
        r1 = self.request_class.from_response(response, formcss="form[action='post.php']")
        fs = _qs(r1)
        self.assertEqual(fs[b'one'], [b'1'])

        r1 = self.request_class.from_response(response, formcss="input[name='four']")
        fs = _qs(r1)
        self.assertEqual(fs[b'three'], [b'3'])

        self.assertRaises(ValueError, self.request_class.from_response,
                          response, formcss="input[name='abc']")


def _buildresponse(body, **kwargs):
    kwargs.setdefault('body', body)
    kwargs.setdefault('url', 'http://example.com')
    kwargs.setdefault('encoding', 'utf-8')
    return HtmlResponse(**kwargs)


def _qs(req, encoding='utf-8', to_unicode=False):
    if req.method == 'POST':
        qs = req.body
    else:
        qs = req.url.partition('?')[2]
    if six.PY2:
        uqs = unquote(to_native_str(qs, encoding))
    elif six.PY3:
        uqs = unquote_to_bytes(qs)
    if to_unicode:
        uqs = uqs.decode(encoding)
    return parse_qs(uqs, True)


class TestRequestMetaUnsafeUrl(MixinUnsafeUrl, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


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


class TestRequestMetaDefault(MixinDefault, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SCRAPY_DEFAULT}


class SubselectorLoaderTest(unittest.TestCase):
    response = HtmlResponse(url="", encoding='utf-8', body=b"""
    <html>
    <body>
    <header>
      <div id="id">marta</div>
      <p>paragraph</p>
    </header>
    <footer class="footer">
      <a href="http://www.scrapy.org">homepage</a>
      <img src="/images/logo.png" width="244" height="65" alt="Scrapy">
    </footer>
    </body>
    </html>
    """)

    def test_nested_xpath(self):
        l = NestedItemLoader(response=self.response)
        nl = l.nested_xpath("//header")
        nl.add_xpath('name', 'div/text()')
        nl.add_css('name_div', '#id')
        nl.add_value('name_value', nl.selector.xpath('div[@id = "id"]/text()').extract())

        self.assertEqual(l.get_output_value('name'), [u'marta'])
        self.assertEqual(l.get_output_value('name_div'), [u'<div id="id">marta</div>'])
        self.assertEqual(l.get_output_value('name_value'),  [u'marta'])

        self.assertEqual(l.get_output_value('name'), nl.get_output_value('name'))
        self.assertEqual(l.get_output_value('name_div'), nl.get_output_value('name_div'))
        self.assertEqual(l.get_output_value('name_value'), nl.get_output_value('name_value'))

    def test_nested_css(self):
        l = NestedItemLoader(response=self.response)
        nl = l.nested_css("header")
        nl.add_xpath('name', 'div/text()')
        nl.add_css('name_div', '#id')
        nl.add_value('name_value', nl.selector.xpath('div[@id = "id"]/text()').extract())

        self.assertEqual(l.get_output_value('name'), [u'marta'])
        self.assertEqual(l.get_output_value('name_div'), [u'<div id="id">marta</div>'])
        self.assertEqual(l.get_output_value('name_value'),  [u'marta'])

        self.assertEqual(l.get_output_value('name'), nl.get_output_value('name'))
        self.assertEqual(l.get_output_value('name_div'), nl.get_output_value('name_div'))
        self.assertEqual(l.get_output_value('name_value'), nl.get_output_value('name_value'))

    def test_nested_replace(self):
        l = NestedItemLoader(response=self.response)
        nl1 = l.nested_xpath('//footer')
        nl2 = nl1.nested_xpath('a')

        l.add_xpath('url', '//footer/a/@href')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])
        nl1.replace_xpath('url', 'img/@src')
        self.assertEqual(l.get_output_value('url'), [u'/images/logo.png'])
        nl2.replace_xpath('url', '@href')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])

    def test_nested_ordering(self):
        l = NestedItemLoader(response=self.response)
        nl1 = l.nested_xpath('//footer')
        nl2 = nl1.nested_xpath('a')

        nl1.add_xpath('url', 'img/@src')
        l.add_xpath('url', '//footer/a/@href')
        nl2.add_xpath('url', 'text()')
        l.add_xpath('url', '//footer/a/@href')

        self.assertEqual(l.get_output_value('url'), [
            u'/images/logo.png',
            u'http://www.scrapy.org',
            u'homepage',
            u'http://www.scrapy.org',
        ])

    def test_nested_load_item(self):
        l = NestedItemLoader(response=self.response)
        nl1 = l.nested_xpath('//footer')
        nl2 = nl1.nested_xpath('img')

        l.add_xpath('name', '//header/div/text()')
        nl1.add_xpath('url', 'a/@href')
        nl2.add_xpath('image', '@src')

        item = l.load_item()

        assert item is l.item
        assert item is nl1.item
        assert item is nl2.item

        self.assertEqual(item['name'], [u'marta'])
        self.assertEqual(item['url'], [u'http://www.scrapy.org'])
        self.assertEqual(item['image'], [u'/images/logo.png'])


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


class DeferUtilsTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_process_chain(self):
        x = yield process_chain([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (cb1 res v1 v2) v1 v2) v1 v2)")

        gotexc = False
        try:
            yield process_chain([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        except TypeError as e:
            gotexc = True
        self.assertTrue(gotexc)

    @defer.inlineCallbacks
    def test_process_chain_both(self):
        x = yield process_chain_both([cb_fail, cb2, cb3], [None, eb1, None], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (eb1 TypeError v1 v2) v1 v2)")

        fail = Failure(ZeroDivisionError())
        x = yield process_chain_both([eb1, cb2, cb3], [eb1, None, None], fail, 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (eb1 ZeroDivisionError v1 v2) v1 v2) v1 v2)")

    @defer.inlineCallbacks
    def test_process_parallel(self):
        x = yield process_parallel([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, ['(cb1 res v1 v2)', '(cb2 res v1 v2)', '(cb3 res v1 v2)'])

    def test_process_parallel_failure(self):
        d = process_parallel([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        self.failUnlessFailure(d, TypeError)
        return d


class FilesPipelineTestCaseFields(unittest.TestCase):

    def test_item_fields_default(self):
        class TestItem(Item):
            name = Field()
            file_urls = Field()
            files = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/files/1.txt'
            item = cls({'name': 'item1', 'file_urls': [url]})
            pipeline = FilesPipeline.from_settings(Settings({'FILES_STORE': 's3://example/files/'}))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['files'], [results[0][1]])

    def test_item_fields_override_settings(self):
        class TestItem(Item):
            name = Field()
            files = Field()
            stored_file = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/files/1.txt'
            item = cls({'name': 'item1', 'files': [url]})
            pipeline = FilesPipeline.from_settings(Settings({
                'FILES_STORE': 's3://example/files/',
                'FILES_URLS_FIELD': 'files',
                'FILES_RESULT_FIELD': 'stored_file'
            }))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['stored_file'], [results[0][1]])


class TestDupeFilterSpider(TestSpider):
    def start_requests(self):
        return (Request(url) for url in self.start_urls)  # no dont_filter=True


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


class XmlItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return XmlItemExporter(self.output, **kwargs)

    def assertXmlEquivalent(self, first, second, msg=None):
        def xmltuple(elem):
            children = list(elem.iterchildren())
            if children:
                return [(child.tag, sorted(xmltuple(child)))
                        for child in children]
            else:
                return [(elem.tag, [(elem.text, ())])]
        def xmlsplit(xmlcontent):
            doc = lxml.etree.fromstring(xmlcontent)
            return xmltuple(doc)
        return self.assertEqual(xmlsplit(first), xmlsplit(second), msg)

    def assertExportResult(self, item, expected_value):
        fp = BytesIO()
        ie = XmlItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        self.assertXmlEquivalent(fp.getvalue(), expected_value)

    def _check_output(self):
        expected_value = b'<?xml version="1.0" encoding="utf-8"?>\n<items><item><age>22</age><name>John\xc2\xa3</name></item></items>'
        self.assertXmlEquivalent(self.output.getvalue(), expected_value)

    def test_multivalued_fields(self):
        self.assertExportResult(
            TestItem(name=[u'John\xa3', u'Doe']),
            b'<?xml version="1.0" encoding="utf-8"?>\n<items><item><name><value>John\xc2\xa3</value><value>Doe</value></name></item></items>'
        )

    def test_nested_item(self):
        i1 = TestItem(name=u'foo\xa3hoo', age='22')
        i2 = dict(name=u'bar', age=i1)
        i3 = TestItem(name=u'buz', age=i2)

        self.assertExportResult(i3,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
                b'<item>'
                    b'<age>'
                        b'<age>'
                            b'<age>22</age>'
                            b'<name>foo\xc2\xa3hoo</name>'
                        b'</age>'
                        b'<name>bar</name>'
                    b'</age>'
                    b'<name>buz</name>'
                b'</item>'
            b'</items>'
        )

    def test_nested_list_item(self):
        i1 = TestItem(name=u'foo')
        i2 = dict(name=u'bar', v2={"egg": ["spam"]})
        i3 = TestItem(name=u'buz', age=[i1, i2])

        self.assertExportResult(i3,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
                b'<item>'
                    b'<age>'
                        b'<value><name>foo</name></value>'
                        b'<value><name>bar</name><v2><egg><value>spam</value></egg></v2></value>'
                    b'</age>'
                    b'<name>buz</name>'
                b'</item>'
            b'</items>'
        )

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        self.assertExportResult(item,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
               b'<item>'
                   b'<float>3.14</float>'
                   b'<boolean>False</boolean>'
                   b'<number>22</number>'
                   b'<time>2015-01-01 01:01:01</time>'
               b'</item>'
            b'</items>'
        )


class ChunkSize3PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 3

class TestSettingsPolicyByName(TestCase):

    def test_valid_name(self):
        for s, p in [
                (POLICY_SCRAPY_DEFAULT, DefaultReferrerPolicy),
                (POLICY_NO_REFERRER, NoReferrerPolicy),
                (POLICY_NO_REFERRER_WHEN_DOWNGRADE, NoReferrerWhenDowngradePolicy),
                (POLICY_SAME_ORIGIN, SameOriginPolicy),
                (POLICY_ORIGIN, OriginPolicy),
                (POLICY_STRICT_ORIGIN, StrictOriginPolicy),
                (POLICY_ORIGIN_WHEN_CROSS_ORIGIN, OriginWhenCrossOriginPolicy),
                (POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN, StrictOriginWhenCrossOriginPolicy),
                (POLICY_UNSAFE_URL, UnsafeUrlPolicy),
            ]:
            settings = Settings({'REFERRER_POLICY': s})
            mw = RefererMiddleware(settings)
            self.assertEqual(mw.default_policy, p)

    def test_valid_name_casevariants(self):
        for s, p in [
                (POLICY_SCRAPY_DEFAULT, DefaultReferrerPolicy),
                (POLICY_NO_REFERRER, NoReferrerPolicy),
                (POLICY_NO_REFERRER_WHEN_DOWNGRADE, NoReferrerWhenDowngradePolicy),
                (POLICY_SAME_ORIGIN, SameOriginPolicy),
                (POLICY_ORIGIN, OriginPolicy),
                (POLICY_STRICT_ORIGIN, StrictOriginPolicy),
                (POLICY_ORIGIN_WHEN_CROSS_ORIGIN, OriginWhenCrossOriginPolicy),
                (POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN, StrictOriginWhenCrossOriginPolicy),
                (POLICY_UNSAFE_URL, UnsafeUrlPolicy),
            ]:
            settings = Settings({'REFERRER_POLICY': s.upper()})
            mw = RefererMiddleware(settings)
            self.assertEqual(mw.default_policy, p)

    def test_invalid_name(self):
        settings = Settings({'REFERRER_POLICY': 'some-custom-unknown-policy'})
        with self.assertRaises(RuntimeError):
            mw = RefererMiddleware(settings)


class Https11WrongHostnameTestCase(Http11TestCase):
    scheme = 'https'

    # above tests use a server certificate for "localhost",
    # client connection to "localhost" too.
    # here we test that even if the server certificate is for another domain,
    # "www.example.com" in this case,
    # the tests still pass
    keyfile = 'keys/example-com.key.pem'
    certfile = 'keys/example-com.cert.pem'


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


class UtilsSpidersTestCase(unittest.TestCase):

    def test_iterate_spider_output(self):
        i = BaseItem()
        r = Request('http://scrapytest.org')
        o = object()

        self.assertEqual(list(iterate_spider_output(i)), [i])
        self.assertEqual(list(iterate_spider_output(r)), [r])
        self.assertEqual(list(iterate_spider_output(o)), [o])
        self.assertEqual(list(iterate_spider_output([r, i, o])), [r, i, o])

    def test_iter_spider_classes(self):
        import tests.test_utils_spider
        it = iter_spider_classes(tests.test_utils_spider)
        self.assertEqual(set(it), {MySpider1, MySpider2})

if __name__ == "__main__":
    unittest.main()


class TestRequestMetaSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SAME_ORIGIN}


class ChunkSize1PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 1

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


class InitSpiderTest(SpiderTest):

    spider_class = InitSpider


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


class TestSettingsStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.StrictOriginWhenCrossOriginPolicy'}


class Http10ProxyTestCase(HttpProxyTestCase):
    download_handler_cls = HTTP10DownloadHandler


class MixinDefault(object):
    """
    Based on https://www.w3.org/TR/referrer-policy/#referrer-policy-no-referrer-when-downgrade

    with some additional filtering of s3://
    """
    scenarii = [
        ('https://example.com/',    'https://scrapy.org/',  b'https://example.com/'),
        ('http://example.com/',     'http://scrapy.org/',   b'http://example.com/'),
        ('http://example.com/',     'https://scrapy.org/',  b'http://example.com/'),
        ('https://example.com/',    'http://scrapy.org/',   None),

        # no credentials leak
        ('http://user:password@example.com/',  'https://scrapy.org/', b'http://example.com/'),

        # no referrer leak for local schemes
        ('file:///home/path/to/somefile.html',  'https://scrapy.org/', None),
        ('file:///home/path/to/somefile.html',  'http://scrapy.org/',  None),

        # no referrer leak for s3 origins
        ('s3://mybucket/path/to/data.csv',  'https://scrapy.org/', None),
        ('s3://mybucket/path/to/data.csv',  'http://scrapy.org/',  None),
    ]


class TestSettingsNoReferrer(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerPolicy'}

