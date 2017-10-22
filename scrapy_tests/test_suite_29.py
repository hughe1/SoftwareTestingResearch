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


class XmlResponseTest(TextResponseTest):

    response_class = XmlResponse

    def test_xml_encoding(self):
        body = b"<xml></xml>"
        r1 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r1, self.response_class._DEFAULT_ENCODING, body)

        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        r2 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r2, 'iso-8859-1', body)

        # make sure replace() preserves the explicit encoding passed in the constructor
        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        r3 = self.response_class("http://www.example.com", body=body, encoding='utf-8')
        body2 = b"New body"
        r4 = r3.replace(body=body2)
        self._assert_response_values(r4, 'utf-8', body2)

    def test_replace_encoding(self):
        # make sure replace() keeps the previous encoding unless overridden explicitly
        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        body2 = b"""<?xml version="1.0" encoding="utf-8"?><xml></xml>"""
        r5 = self.response_class("http://www.example.com", body=body)
        r6 = r5.replace(body=body2)
        r7 = r5.replace(body=body2, encoding='utf-8')
        self._assert_response_values(r5, 'iso-8859-1', body)
        self._assert_response_values(r6, 'iso-8859-1', body2)
        self._assert_response_values(r7, 'utf-8', body2)

    def test_selector(self):
        body = b'<?xml version="1.0" encoding="utf-8"?><xml><elem>value</elem></xml>'
        response = self.response_class("http://www.example.com", body=body)

        self.assertIsInstance(response.selector, Selector)
        self.assertEqual(response.selector.type, 'xml')
        self.assertIs(response.selector, response.selector)  # property is cached
        self.assertIs(response.selector.response, response)

        self.assertEqual(
            response.selector.xpath("//elem/text()").extract(),
            [u'value']
        )

    def test_selector_shortcuts(self):
        body = b'<?xml version="1.0" encoding="utf-8"?><xml><elem>value</elem></xml>'
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//elem/text()").extract(),
            response.selector.xpath("//elem/text()").extract(),
        )

    def test_selector_shortcuts_kwargs(self):
        body = b'''<?xml version="1.0" encoding="utf-8"?>
        <xml xmlns:somens="http://scrapy.org">
        <somens:elem>value</somens:elem>
        </xml>'''
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//s:elem/text()", namespaces={'s': 'http://scrapy.org'}).extract(),
            response.selector.xpath("//s:elem/text()", namespaces={'s': 'http://scrapy.org'}).extract(),
        )

        response.selector.register_namespace('s2', 'http://scrapy.org')
        self.assertEqual(
            response.xpath("//s1:elem/text()", namespaces={'s1': 'http://scrapy.org'}).extract(),
            response.selector.xpath("//s2:elem/text()").extract(),
        )

class ImagesPipelineTestCase(unittest.TestCase):

    skip = skip

    def setUp(self):
        self.tempdir = mkdtemp()
        self.pipeline = ImagesPipeline(self.tempdir, download_func=_mocked_download_func)

    def tearDown(self):
        rmtree(self.tempdir)

    def test_file_path(self):
        file_path = self.pipeline.file_path
        self.assertEqual(file_path(Request("https://dev.mydeco.com/mydeco.gif")),
                         'full/3fd165099d8e71b8a48b2683946e64dbfad8b52d.jpg')
        self.assertEqual(file_path(Request("http://www.maddiebrown.co.uk///catalogue-items//image_54642_12175_95307.jpg")),
                         'full/0ffcd85d563bca45e2f90becd0ca737bc58a00b2.jpg')
        self.assertEqual(file_path(Request("https://dev.mydeco.com/two/dirs/with%20spaces%2Bsigns.gif")),
                         'full/b250e3a74fff2e4703e310048a5b13eba79379d2.jpg')
        self.assertEqual(file_path(Request("http://www.dfsonline.co.uk/get_prod_image.php?img=status_0907_mdm.jpg")),
                         'full/4507be485f38b0da8a0be9eb2e1dfab8a19223f2.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532/")),
                         'full/97ee6f8a46cbbb418ea91502fd24176865cf39b2.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532")),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532"),
                                   response=Response("http://www.dorma.co.uk/images/product_details/2532"),
                                   info=object()),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1.jpg')

    def test_thumbnail_name(self):
        thumb_path = self.pipeline.thumb_path
        name = '50'
        self.assertEqual(thumb_path(Request("file:///tmp/foo.jpg"), name),
                         'thumbs/50/38a86208c36e59d4404db9e37ce04be863ef0335.jpg')
        self.assertEqual(thumb_path(Request("file://foo.png"), name),
                         'thumbs/50/e55b765eba0ec7348e50a1df496040449071b96a.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/foo"), name),
                         'thumbs/50/0329ad83ebb8e93ea7c7906d46e9ed55f7349a50.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/some.name/foo"), name),
                         'thumbs/50/850233df65a5b83361798f532f1fc549cd13cbe9.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/some.name/foo"), name,
                                    response=Response("file:///tmp/some.name/foo"),
                                    info=object()),
                         'thumbs/50/850233df65a5b83361798f532f1fc549cd13cbe9.jpg')

    def test_convert_image(self):
        SIZE = (100, 100)
        # straigh forward case: RGB and JPEG
        COLOUR = (0, 127, 255)
        im = _create_image('JPEG', 'RGB', SIZE, COLOUR)
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, COLOUR)])

        # check that thumbnail keep image ratio
        thumbnail, _ = self.pipeline.convert_image(converted, size=(10, 25))
        self.assertEqual(thumbnail.mode, 'RGB')
        self.assertEqual(thumbnail.size, (10, 10))

        # transparency case: RGBA and PNG
        COLOUR = (0, 127, 255, 50)
        im = _create_image('PNG', 'RGBA', SIZE, COLOUR)
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, (205, 230, 255))])

        # transparency case with palette: P and PNG
        COLOUR = (0, 127, 255, 50)
        im = _create_image('PNG', 'RGBA', SIZE, COLOUR)
        im = im.convert('P')
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, (205, 230, 255))])


class FileTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpname = self.mktemp()
        with open(self.tmpname + '^', 'w') as f:
            f.write('0123456789')
        self.download_request = FileDownloadHandler(Settings()).download_request

    def tearDown(self):
        os.unlink(self.tmpname + '^')

    def test_download(self):
        def _test(response):
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.body, b'0123456789')

        request = Request(path_to_file_uri(self.tmpname + '^'))
        assert request.url.upper().endswith('%5E')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_non_existent(self):
        request = Request('file://%s' % self.mktemp())
        d = self.download_request(request, Spider('foo'))
        return self.assertFailure(d, IOError)


class DeprecatedImagesPipeline(ImagesPipeline):
    def file_key(self, url):
        return self.image_key(url)

    def image_key(self, url):
        image_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'empty/%s.jpg' % (image_guid)

    def thumb_key(self, url, thumb_id):
        thumb_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'thumbsup/%s/%s.jpg' % (thumb_id, thumb_guid)


class VersionTest(ProcessTest, unittest.TestCase):

    command = 'version'

    @defer.inlineCallbacks
    def test_output(self):
        encoding = getattr(sys.stdout, 'encoding') or 'utf-8'
        _, out, _ = yield self.execute([])
        self.assertEqual(
            out.strip().decode(encoding),
            "Scrapy %s" % scrapy.__version__,
        )

    @defer.inlineCallbacks
    def test_verbose_output(self):
        encoding = getattr(sys.stdout, 'encoding') or 'utf-8'
        _, out, _ = yield self.execute(['-v'])
        headers = [l.partition(":")[0].strip()
                   for l in out.strip().decode(encoding).splitlines()]
        self.assertEqual(headers, ['Scrapy', 'lxml', 'libxml2',
                                   'cssselect', 'parsel', 'w3lib',
                                   'Twisted', 'Python', 'pyOpenSSL',
                                   'cryptography', 'Platform'])

class TestRequestMetaPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    req_meta = {'referrer_policy': POLICY_NO_REFERRER}


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

class RFC2616PolicyTest(DefaultStorageTest):

    policy_class = 'scrapy.extensions.httpcache.RFC2616Policy'

    def _process_requestresponse(self, mw, request, response):
        result = None
        try:
            result = mw.process_request(request, self.spider)
            if result:
                assert isinstance(result, (Request, Response))
                return result
            else:
                result = mw.process_response(request, response, self.spider)
                assert isinstance(result, Response)
                return result
        except Exception:
            print('Request', request)
            print('Response', response)
            print('Result', result)
            raise

    def test_request_cacheability(self):
        res0 = Response(self.request.url, status=200,
                        headers={'Expires': self.tomorrow})
        req0 = Request('http://example.com')
        req1 = req0.replace(headers={'Cache-Control': 'no-store'})
        req2 = req0.replace(headers={'Cache-Control': 'no-cache'})
        with self._middleware() as mw:
            # response for a request with no-store must not be cached
            res1 = self._process_requestresponse(mw, req1, res0)
            self.assertEqualResponse(res1, res0)
            assert mw.storage.retrieve_response(self.spider, req1) is None
            # Re-do request without no-store and expect it to be cached
            res2 = self._process_requestresponse(mw, req0, res0)
            assert 'cached' not in res2.flags
            res3 = mw.process_request(req0, self.spider)
            assert 'cached' in res3.flags
            self.assertEqualResponse(res2, res3)
            # request with no-cache directive must not return cached response
            # but it allows new response to be stored
            res0b = res0.replace(body=b'foo')
            res4 = self._process_requestresponse(mw, req2, res0b)
            self.assertEqualResponse(res4, res0b)
            assert 'cached' not in res4.flags
            res5 = self._process_requestresponse(mw, req0, None)
            self.assertEqualResponse(res5, res0b)
            assert 'cached' in res5.flags

    def test_response_cacheability(self):
        responses = [
            # 304 is not cacheable no matter what servers sends
            (False, 304, {}),
            (False, 304, {'Last-Modified': self.yesterday}),
            (False, 304, {'Expires': self.tomorrow}),
            (False, 304, {'Etag': 'bar'}),
            (False, 304, {'Cache-Control': 'max-age=3600'}),
            # Always obey no-store cache control
            (False, 200, {'Cache-Control': 'no-store'}),
            (False, 200, {'Cache-Control': 'no-store, max-age=300'}),  # invalid
            (False, 200, {'Cache-Control': 'no-store', 'Expires': self.tomorrow}),  # invalid
            # Ignore responses missing expiration and/or validation headers
            (False, 200, {}),
            (False, 302, {}),
            (False, 307, {}),
            (False, 404, {}),
            # Cache responses with expiration and/or validation headers
            (True, 200, {'Last-Modified': self.yesterday}),
            (True, 203, {'Last-Modified': self.yesterday}),
            (True, 300, {'Last-Modified': self.yesterday}),
            (True, 301, {'Last-Modified': self.yesterday}),
            (True, 308, {'Last-Modified': self.yesterday}),
            (True, 401, {'Last-Modified': self.yesterday}),
            (True, 404, {'Cache-Control': 'public, max-age=600'}),
            (True, 302, {'Expires': self.tomorrow}),
            (True, 200, {'Etag': 'foo'}),
        ]
        with self._middleware() as mw:
            for idx, (shouldcache, status, headers) in enumerate(responses):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                res1 = self._process_requestresponse(mw, req0, res0)
                res304 = res0.replace(status=304)
                res2 = self._process_requestresponse(mw, req0, res304 if shouldcache else res0)
                self.assertEqualResponse(res1, res0)
                self.assertEqualResponse(res2, res0)
                resc = mw.storage.retrieve_response(self.spider, req0)
                if shouldcache:
                    self.assertEqualResponse(resc, res1)
                    assert 'cached' in res2.flags and res2.status != 304
                else:
                    self.assertFalse(resc)
                    assert 'cached' not in res2.flags

        # cache unconditionally unless response contains no-store or is a 304
        with self._middleware(HTTPCACHE_ALWAYS_STORE=True) as mw:
            for idx, (_, status, headers) in enumerate(responses):
                shouldcache = 'no-store' not in headers.get('Cache-Control', '') and status != 304
                req0 = Request('http://example2-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                res1 = self._process_requestresponse(mw, req0, res0)
                res304 = res0.replace(status=304)
                res2 = self._process_requestresponse(mw, req0, res304 if shouldcache else res0)
                self.assertEqualResponse(res1, res0)
                self.assertEqualResponse(res2, res0)
                resc = mw.storage.retrieve_response(self.spider, req0)
                if shouldcache:
                    self.assertEqualResponse(resc, res1)
                    assert 'cached' in res2.flags and res2.status != 304
                else:
                    self.assertFalse(resc)
                    assert 'cached' not in res2.flags

    def test_cached_and_fresh(self):
        sampledata = [
            (200, {'Date': self.yesterday, 'Expires': self.tomorrow}),
            (200, {'Date': self.yesterday, 'Cache-Control': 'max-age=86405'}),
            (200, {'Age': '299', 'Cache-Control': 'max-age=300'}),
            # Obey max-age if present over any others
            (200, {'Date': self.today,
                   'Age': '86405',
                   'Cache-Control': 'max-age=' + str(86400 * 3),
                   'Expires': self.yesterday,
                   'Last-Modified': self.yesterday,
                   }),
            # obey Expires if max-age is not present
            (200, {'Date': self.yesterday,
                   'Age': '86400',
                   'Cache-Control': 'public',
                   'Expires': self.tomorrow,
                   'Last-Modified': self.yesterday,
                   }),
            # Default missing Date header to right now
            (200, {'Expires': self.tomorrow}),
            # Firefox - Expires if age is greater than 10% of (Date - Last-Modified)
            (200, {'Date': self.today, 'Last-Modified': self.yesterday, 'Age': str(86400 / 10 - 1)}),
            # Firefox - Set one year maxage to permanent redirects missing expiration info
            (300, {}), (301, {}), (308, {}),
        ]
        with self._middleware() as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                # cache fresh response
                res1 = self._process_requestresponse(mw, req0, res0)
                self.assertEqualResponse(res1, res0)
                assert 'cached' not in res1.flags
                # return fresh cached response without network interaction
                res2 = self._process_requestresponse(mw, req0, None)
                self.assertEqualResponse(res1, res2)
                assert 'cached' in res2.flags
                # validate cached response if request max-age set as 0
                req1 = req0.replace(headers={'Cache-Control': 'max-age=0'})
                res304 = res0.replace(status=304)
                assert mw.process_request(req1, self.spider) is None
                res3 = self._process_requestresponse(mw, req1, res304)
                self.assertEqualResponse(res1, res3)
                assert 'cached' in res3.flags

    def test_cached_and_stale(self):
        sampledata = [
            (200, {'Date': self.today, 'Expires': self.yesterday}),
            (200, {'Date': self.today, 'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Expires': self.yesterday}),
            (200, {'Expires': self.yesterday, 'ETag': 'foo'}),
            (200, {'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Expires': self.tomorrow, 'Age': '86405'}),
            (200, {'Cache-Control': 'max-age=86400', 'Age': '86405'}),
            # no-cache forces expiration, also revalidation if validators exists
            (200, {'Cache-Control': 'no-cache'}),
            (200, {'Cache-Control': 'no-cache', 'ETag': 'foo'}),
            (200, {'Cache-Control': 'no-cache', 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'no-cache,must-revalidate', 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'must-revalidate', 'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'max-age=86400,must-revalidate', 'Age': '86405'}),
        ]
        with self._middleware() as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0a = Response(req0.url, status=status, headers=headers)
                # cache expired response
                res1 = self._process_requestresponse(mw, req0, res0a)
                self.assertEqualResponse(res1, res0a)
                assert 'cached' not in res1.flags
                # Same request but as cached response is stale a new response must
                # be returned
                res0b = res0a.replace(body=b'bar')
                res2 = self._process_requestresponse(mw, req0, res0b)
                self.assertEqualResponse(res2, res0b)
                assert 'cached' not in res2.flags
                cc = headers.get('Cache-Control', '')
                # Previous response expired too, subsequent request to same
                # resource must revalidate and succeed on 304 if validators
                # are present
                if 'ETag' in headers or 'Last-Modified' in headers:
                    res0c = res0b.replace(status=304)
                    res3 = self._process_requestresponse(mw, req0, res0c)
                    self.assertEqualResponse(res3, res0b)
                    assert 'cached' in res3.flags
                    # get cached response on server errors unless must-revalidate
                    # in cached response
                    res0d = res0b.replace(status=500)
                    res4 = self._process_requestresponse(mw, req0, res0d)
                    if 'must-revalidate' in cc:
                        assert 'cached' not in res4.flags
                        self.assertEqualResponse(res4, res0d)
                    else:
                        assert 'cached' in res4.flags
                        self.assertEqualResponse(res4, res0b)
                # Requests with max-stale can fetch expired cached responses
                # unless cached response has must-revalidate
                req1 = req0.replace(headers={'Cache-Control': 'max-stale'})
                res5 = self._process_requestresponse(mw, req1, res0b)
                self.assertEqualResponse(res5, res0b)
                if 'no-cache' in cc or 'must-revalidate' in cc:
                    assert 'cached' not in res5.flags
                else:
                    assert 'cached' in res5.flags

    def test_process_exception(self):
        with self._middleware() as mw:
            res0 = Response(self.request.url, headers={'Expires': self.yesterday})
            req0 = Request(self.request.url)
            self._process_requestresponse(mw, req0, res0)
            for e in mw.DOWNLOAD_EXCEPTIONS:
                # Simulate encountering an error on download attempts
                assert mw.process_request(req0, self.spider) is None
                res1 = mw.process_exception(req0, e('foo'), self.spider)
                # Use cached response as recovery
                assert 'cached' in res1.flags
                self.assertEqualResponse(res0, res1)
            # Do not use cached response for unhandled exceptions
            mw.process_request(req0, self.spider)
            assert mw.process_exception(req0, Exception('foo'), self.spider) is None

    def test_ignore_response_cache_controls(self):
        sampledata = [
            (200, {'Date': self.yesterday, 'Expires': self.tomorrow}),
            (200, {'Date': self.yesterday, 'Cache-Control': 'no-store,max-age=86405'}),
            (200, {'Age': '299', 'Cache-Control': 'max-age=300,no-cache'}),
            (300, {'Cache-Control': 'no-cache'}),
            (200, {'Expires': self.tomorrow, 'Cache-Control': 'no-store'}),
        ]
        with self._middleware(HTTPCACHE_IGNORE_RESPONSE_CACHE_CONTROLS=['no-cache', 'no-store']) as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                # cache fresh response
                res1 = self._process_requestresponse(mw, req0, res0)
                self.assertEqualResponse(res1, res0)
                assert 'cached' not in res1.flags
                # return fresh cached response without network interaction
                res2 = self._process_requestresponse(mw, req0, None)
                self.assertEqualResponse(res1, res2)
                assert 'cached' in res2.flags

if __name__ == '__main__':
    unittest.main()

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


class PickleFifoDiskQueueTest(MarshalFifoDiskQueueTest):

    chunksize = 100000

    def queue(self):
        return PickleFifoDiskQueue(self.qpath, chunksize=self.chunksize)

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

class TestOffsiteMiddleware4(TestOffsiteMiddleware3):

    def _get_spider(self):
      bad_hostname = urlparse('http:////scrapytest.org').hostname
      return dict(name='foo', allowed_domains=['scrapytest.org', None, bad_hostname])

    def test_process_spider_output(self):
      res = Response('http://scrapytest.org')
      reqs = [Request('http://scrapytest.org/1')]
      out = list(self.mw.process_spider_output(res, reqs, self.spider))
      self.assertEqual(out, reqs)

class M3(object):

    def process(self, response, request, spider):
        pass


class TestSettingsNoReferrerWhenDowngrade(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}


class DbmStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.DbmCacheStorage'


class Foo(trackref.object_ref):
    pass


class ContentLengthHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of the Content-Length
    header from the request.
    """
    def render(self, request):
        return request.requestHeaders.getRawHeaders(b"content-length")[0]


class DeprecatedFilesPipeline(FilesPipeline):
    def file_key(self, url):
        media_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        media_ext = os.path.splitext(url)[1]
        return 'empty/%s%s' % (media_guid, media_ext)


class MyWarning(UserWarning):
    pass


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


class UrlparseTestCase(unittest.TestCase):

    def test_s3_url(self):
        p = urlparse('s3://bucket/key/name?param=value')
        self.assertEqual(p.scheme, 's3')
        self.assertEqual(p.hostname, 'bucket')
        self.assertEqual(p.path, '/key/name')
        self.assertEqual(p.query, 'param=value')

class TestHttpErrorMiddlewareIntegrational(TrialTestCase):
    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_middleware_works(self):
        crawler = get_crawler(_HttpErrorSpider)
        yield crawler.crawl()
        assert not crawler.spider.skipped, crawler.spider.skipped
        self.assertEqual(crawler.spider.parsed, {'200'})
        self.assertEqual(crawler.spider.failed, {'404', '402', '500'})

        get_value = crawler.stats.get_value
        self.assertEqual(get_value('httperror/response_ignored_count'), 3)
        self.assertEqual(get_value('httperror/response_ignored_status_count/404'), 1)
        self.assertEqual(get_value('httperror/response_ignored_status_count/402'), 1)
        self.assertEqual(get_value('httperror/response_ignored_status_count/500'), 1)

    @defer.inlineCallbacks
    def test_logging(self):
        crawler = get_crawler(_HttpErrorSpider)
        with LogCapture() as log:
            yield crawler.crawl(bypass_status_codes={402})
        self.assertEqual(crawler.spider.parsed, {'200', '402'})
        self.assertEqual(crawler.spider.skipped, {'402'})
        self.assertEqual(crawler.spider.failed, {'404', '500'})

        self.assertIn('Ignoring response <404', str(log))
        self.assertIn('Ignoring response <500', str(log))
        self.assertNotIn('Ignoring response <200', str(log))
        self.assertNotIn('Ignoring response <402', str(log))

    @defer.inlineCallbacks
    def test_logging_level(self):
        # HttpError logs ignored responses with level INFO
        crawler = get_crawler(_HttpErrorSpider)
        with LogCapture(level=logging.INFO) as log:
            yield crawler.crawl()
        self.assertEqual(crawler.spider.parsed, {'200'})
        self.assertEqual(crawler.spider.failed, {'404', '402', '500'})

        self.assertIn('Ignoring response <402', str(log))
        self.assertIn('Ignoring response <404', str(log))
        self.assertIn('Ignoring response <500', str(log))
        self.assertNotIn('Ignoring response <200', str(log))

        # with level WARNING, we shouldn't capture anything from HttpError
        crawler = get_crawler(_HttpErrorSpider)
        with LogCapture(level=logging.WARNING) as log:
            yield crawler.crawl()
        self.assertEqual(crawler.spider.parsed, {'200'})
        self.assertEqual(crawler.spider.failed, {'404', '402', '500'})

        self.assertNotIn('Ignoring response <402', str(log))
        self.assertNotIn('Ignoring response <404', str(log))
        self.assertNotIn('Ignoring response <500', str(log))
        self.assertNotIn('Ignoring response <200', str(log))

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
class TestSettingsOrigin(MixinOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginPolicy'}


class TestRequestMetaPredecence003(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


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


class BaseMediaPipelineTestCase(unittest.TestCase):

    pipeline_class = MediaPipeline
    settings = None

    def setUp(self):
        self.spider = Spider('media.com')
        self.pipe = self.pipeline_class(download_func=_mocked_download_func,
                                        settings=Settings(self.settings))
        self.pipe.open_spider(self.spider)
        self.info = self.pipe.spiderinfo

    def tearDown(self):
        for name, signal in vars(signals).items():
            if not name.startswith('_'):
                disconnect_all(signal)

    def test_default_media_to_download(self):
        request = Request('http://url')
        assert self.pipe.media_to_download(request, self.info) is None

    def test_default_get_media_requests(self):
        item = dict(name='name')
        assert self.pipe.get_media_requests(item, self.info) is None

    def test_default_media_downloaded(self):
        request = Request('http://url')
        response = Response('http://url', body=b'')
        assert self.pipe.media_downloaded(response, request, self.info) is response

    def test_default_media_failed(self):
        request = Request('http://url')
        fail = Failure(Exception())
        assert self.pipe.media_failed(fail, request, self.info) is fail

    def test_default_item_completed(self):
        item = dict(name='name')
        assert self.pipe.item_completed([], item, self.info) is item

        # Check that failures are logged by default
        fail = Failure(Exception())
        results = [(True, 1), (False, fail)]

        with LogCapture() as l:
            new_item = self.pipe.item_completed(results, item, self.info)

        assert new_item is item
        assert len(l.records) == 1
        record = l.records[0]
        assert record.levelname == 'ERROR'
        self.assertTupleEqual(record.exc_info, failure_to_exc_info(fail))

        # disable failure logging and check again
        self.pipe.LOG_FAILED_RESULTS = False
        with LogCapture() as l:
            new_item = self.pipe.item_completed(results, item, self.info)
        assert new_item is item
        assert len(l.records) == 0

    @inlineCallbacks
    def test_default_process_item(self):
        item = dict(name='name')
        new_item = yield self.pipe.process_item(item, self.spider)
        assert new_item is item

    def test_modify_media_request(self):
        request = Request('http://url')
        self.pipe._modify_media_request(request)
        assert request.meta == {'handle_httpstatus_all': True}


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


class DefaultStorageTest(_BaseTest):

    def test_storage(self):
        with self._storage() as storage:
            request2 = self.request.copy()
            assert storage.retrieve_response(self.spider, request2) is None

            storage.store_response(self.spider, self.request, self.response)
            response2 = storage.retrieve_response(self.spider, request2)
            assert isinstance(response2, HtmlResponse)  # content-type header
            self.assertEqualResponse(self.response, response2)

            time.sleep(2)  # wait for cache to expire
            assert storage.retrieve_response(self.spider, request2) is None

    def test_storage_never_expire(self):
        with self._storage(HTTPCACHE_EXPIRATION_SECS=0) as storage:
            assert storage.retrieve_response(self.spider, self.request) is None
            storage.store_response(self.spider, self.request, self.response)
            time.sleep(0.5)  # give the chance to expire
            assert storage.retrieve_response(self.spider, self.request)


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


class TestSpider(Spider):
    name = 'demo_spider'

    def returns_request(self, response):
        """ method which returns request
        @url http://scrapy.org
        @returns requests 1
        """
        return Request('http://scrapy.org', callback=self.returns_item)

    def returns_item(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 1 1
        """
        return TestItem(url=response.url)

    def returns_dict_item(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 1 1
        """
        return {"url": response.url}

    def returns_fail(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 0 0
        """
        return TestItem(url=response.url)

    def returns_dict_fail(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 0 0
        """
        return {'url': response.url}

    def scrapes_item_ok(self, response):
        """ returns item with name and url
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return TestItem(name='test', url=response.url)

    def scrapes_dict_item_ok(self, response):
        """ returns item with name and url
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return {'name': 'test', 'url': response.url}

    def scrapes_item_fail(self, response):
        """ returns item with no name
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return TestItem(url=response.url)

    def scrapes_dict_item_fail(self, response):
        """ returns item with no name
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return {'url': response.url}

    def parse_no_url(self, response):
        """ method with no url
        @returns items 1 1
        """
        pass

