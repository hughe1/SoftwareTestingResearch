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


class Https11InvalidDNSId(Https11TestCase):
    """Connect to HTTPS hosts with IP while certificate uses domain names IDs."""

    def setUp(self):
        super(Https11InvalidDNSId, self).setUp()
        self.host = '127.0.0.1'


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


class FileDownloadCrawlTestCase(TestCase):
    pipeline_class = 'scrapy.pipelines.files.FilesPipeline'
    store_setting_key = 'FILES_STORE'
    media_key = 'files'
    media_urls_key = 'file_urls'
    expected_checksums = set([
        '5547178b89448faf0015a13f904c936e',
        'c2281c83670e31d8aaab7cb642b824db',
        'ed3f6538dc15d4d9179dae57319edc5f'])

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

        # prepare a directory for storing files
        self.tmpmediastore = self.mktemp()
        os.mkdir(self.tmpmediastore)
        self.settings = {
            'ITEM_PIPELINES': {self.pipeline_class: 1},
            self.store_setting_key: self.tmpmediastore,
        }
        self.runner = CrawlerRunner(self.settings)
        self.items = []

    def tearDown(self):
        shutil.rmtree(self.tmpmediastore)
        self.items = []
        self.mockserver.__exit__(None, None, None)

    def _on_item_scraped(self, item):
        self.items.append(item)

    def _create_crawler(self, spider_class, **kwargs):
        crawler = self.runner.create_crawler(spider_class, **kwargs)
        crawler.signals.connect(self._on_item_scraped, signals.item_scraped)
        return crawler

    def _assert_files_downloaded(self, items, logs):
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])

        # check that logs show the expected number of successful file downloads
        file_dl_success = 'File (downloaded): Downloaded file from'
        self.assertEqual(logs.count(file_dl_success), 3)

        # check that the images/files checksums are what we know they should be
        if self.expected_checksums is not None:
            checksums = set(
                i['checksum']
                    for item in items
                        for i in item[self.media_key])
            self.assertEqual(checksums, self.expected_checksums)

        # check that the image files where actually written to the media store
        for item in items:
            for i in item[self.media_key]:
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.tmpmediastore, i['path'])))

    def _assert_files_download_failure(self, crawler, items, code, logs):

        # check that the item does NOT have the "images/files" field populated
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])
        self.assertFalse(items[0][self.media_key])

        # check that there was 1 successful fetch and 3 other responses with non-200 code
        self.assertEqual(crawler.stats.get_value('downloader/request_method_count/GET'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_count'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/200'), 1)
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/%d' % code), 3)

        # check that logs do show the failure on the file downloads
        file_dl_failure = 'File (code: %d): Error downloading file from' % code
        self.assertEqual(logs.count(file_dl_failure), 3)

        # check that no files were written to the media store
        self.assertEqual(os.listdir(self.tmpmediastore), [])

    @defer.inlineCallbacks
    def test_download_media(self):
        crawler = self._create_crawler(MediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_downloaded(self.items, str(log))

    @defer.inlineCallbacks
    def test_download_media_wrong_urls(self):
        crawler = self._create_crawler(BrokenLinksMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_download_failure(crawler, self.items, 404, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_default_failure(self):
        crawler = self._create_crawler(RedirectedMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_download_failure(crawler, self.items, 302, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_allowed(self):
        settings = dict(self.settings)
        settings.update({'MEDIA_ALLOW_REDIRECTS': True})
        self.runner = CrawlerRunner(settings)

        crawler = self._create_crawler(RedirectedMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_downloaded(self.items, str(log))
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/302'), 3)


class GenspiderCommandTest(CommandTest):

    def test_arguments(self):
        # only pass one argument. spider script shouldn't be created
        self.assertEqual(2, self.call('genspider', 'test_name'))
        assert not exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))
        # pass two arguments <name> <domain>. spider script should be created
        self.assertEqual(0, self.call('genspider', 'test_name', 'test.com'))
        assert exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))

    def test_template(self, tplname='crawl'):
        args = ['--template=%s' % tplname] if tplname else []
        spname = 'test_spider'
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Created spider %r using template %r in module" % (spname, tplname), out)
        self.assertTrue(exists(join(self.proj_mod_path, 'spiders', 'test_spider.py')))
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Spider %r already exists in module" % spname, out)

    def test_template_basic(self):
        self.test_template('basic')

    def test_template_csvfeed(self):
        self.test_template('csvfeed')

    def test_template_xmlfeed(self):
        self.test_template('xmlfeed')

    def test_list(self):
        self.assertEqual(0, self.call('genspider', '--list'))

    def test_dump(self):
        self.assertEqual(0, self.call('genspider', '--dump=basic'))
        self.assertEqual(0, self.call('genspider', '-d', 'basic'))

    def test_same_name_as_project(self):
        self.assertEqual(2, self.call('genspider', self.project_name))
        assert not exists(join(self.proj_mod_path, 'spiders', '%s.py' % self.project_name))


class DeprecatedHttpTestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HttpDownloadHandler


class TestItem(NameItem):
    url = Field()
    summary = Field()


class MySpider(scrapy.Spider):
    name = '{0}'

    def parse(self, response):
        if getattr(self, 'test_arg', None):
            self.logger.debug('It Works!')
        return [scrapy.Item(), dict(foo='bar')]


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


class XmliterTestCase(unittest.TestCase):

    xmliter = staticmethod(xmliter)

    def test_xmliter(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>\
            <products xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="someschmea.xsd">\
              <product id="001">\
                <type>Type 1</type>\
                <name>Name 1</name>\
              </product>\
              <product id="002">\
                <type>Type 2</type>\
                <name>Name 2</name>\
              </product>\
            </products>"""

        response = XmlResponse(url="http://example.com", body=body)
        attrs = []
        for x in self.xmliter(response, 'product'):
            attrs.append((x.xpath("@id").extract(), x.xpath("name/text()").extract(), x.xpath("./type/text()").extract()))

        self.assertEqual(attrs,
                         [(['001'], ['Name 1'], ['Type 1']), (['002'], ['Name 2'], ['Type 2'])])

    def test_xmliter_unusual_node(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>
            <root>
                <matchme...></matchme...>
                <matchmenot></matchmenot>
            </root>
        """
        response = XmlResponse(url="http://example.com", body=body)
        nodenames = [e.xpath('name()').extract()
                 for e in self.xmliter(response, 'matchme...')]
        self.assertEqual(nodenames, [['matchme...']])

    def test_xmliter_unicode(self):
        # example taken from https://github.com/scrapy/scrapy/issues/1665
        body = u"""<?xml version="1.0" encoding="UTF-8"?>
            <þingflokkar>
               <þingflokkur id="26">
                  <heiti />
                  <skammstafanir>
                     <stuttskammstöfun>-</stuttskammstöfun>
                     <löngskammstöfun />
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>80</fyrstaþing>
                  </tímabil>
               </þingflokkur>
               <þingflokkur id="21">
                  <heiti>Alþýðubandalag</heiti>
                  <skammstafanir>
                     <stuttskammstöfun>Ab</stuttskammstöfun>
                     <löngskammstöfun>Alþb.</löngskammstöfun>
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>76</fyrstaþing>
                     <síðastaþing>123</síðastaþing>
                  </tímabil>
               </þingflokkur>
               <þingflokkur id="27">
                  <heiti>Alþýðuflokkur</heiti>
                  <skammstafanir>
                     <stuttskammstöfun>A</stuttskammstöfun>
                     <löngskammstöfun>Alþfl.</löngskammstöfun>
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>27</fyrstaþing>
                     <síðastaþing>120</síðastaþing>
                  </tímabil>
               </þingflokkur>
            </þingflokkar>"""

        for r in (
            # with bytes
            XmlResponse(url="http://example.com", body=body.encode('utf-8')),
            # Unicode body needs encoding information
            XmlResponse(url="http://example.com", body=body, encoding='utf-8')):

            attrs = []
            for x in self.xmliter(r, u'þingflokkur'):
                attrs.append((x.xpath('@id').extract(),
                              x.xpath(u'./skammstafanir/stuttskammstöfun/text()').extract(),
                              x.xpath(u'./tímabil/fyrstaþing/text()').extract()))

            self.assertEqual(attrs,
                             [([u'26'], [u'-'], [u'80']),
                              ([u'21'], [u'Ab'], [u'76']),
                              ([u'27'], [u'A'], [u'27'])])

    def test_xmliter_text(self):
        body = u"""<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>"""

        self.assertEqual([x.xpath("text()").extract() for x in self.xmliter(body, 'product')],
                         [[u'one'], [u'two']])

    def test_xmliter_namespaces(self):
        body = b"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
                <channel>
                <title>My Dummy Company</title>
                <link>http://www.mydummycompany.com</link>
                <description>This is a dummy company. We do nothing.</description>
                <item>
                    <title>Item 1</title>
                    <description>This is item 1</description>
                    <link>http://www.mydummycompany.com/items/1</link>
                    <g:image_link>http://www.mydummycompany.com/images/item1.jpg</g:image_link>
                    <g:id>ITEM_1</g:id>
                    <g:price>400</g:price>
                </item>
                </channel>
            </rss>
        """
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'item')

        node = next(my_iter)
        node.register_namespace('g', 'http://base.google.com/ns/1.0')
        self.assertEqual(node.xpath('title/text()').extract(), ['Item 1'])
        self.assertEqual(node.xpath('description/text()').extract(), ['This is item 1'])
        self.assertEqual(node.xpath('link/text()').extract(), ['http://www.mydummycompany.com/items/1'])
        self.assertEqual(node.xpath('g:image_link/text()').extract(), ['http://www.mydummycompany.com/images/item1.jpg'])
        self.assertEqual(node.xpath('g:id/text()').extract(), ['ITEM_1'])
        self.assertEqual(node.xpath('g:price/text()').extract(), ['400'])
        self.assertEqual(node.xpath('image_link/text()').extract(), [])
        self.assertEqual(node.xpath('id/text()').extract(), [])
        self.assertEqual(node.xpath('price/text()').extract(), [])

    def test_xmliter_exception(self):
        body = u"""<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>"""

        iter = self.xmliter(body, 'product')
        next(iter)
        next(iter)

        self.assertRaises(StopIteration, next, iter)

    def test_xmliter_objtype_exception(self):
        i = self.xmliter(42, 'product')
        self.assertRaises(AssertionError, next, i)

    def test_xmliter_encoding(self):
        body = b'<?xml version="1.0" encoding="ISO-8859-9"?>\n<xml>\n    <item>Some Turkish Characters \xd6\xc7\xde\xdd\xd0\xdc \xfc\xf0\xfd\xfe\xe7\xf6</item>\n</xml>\n\n'
        response = XmlResponse('http://www.example.com', body=body)
        self.assertEqual(
            next(self.xmliter(response, 'item')).extract(),
            u'<item>Some Turkish Characters \xd6\xc7\u015e\u0130\u011e\xdc \xfc\u011f\u0131\u015f\xe7\xf6</item>'
        )


class HttpobjUtilsTest(unittest.TestCase):

    def test_urlparse_cached(self):
        url = "http://www.example.com/index.html"
        request1 = Request(url)
        request2 = Request(url)
        req1a = urlparse_cached(request1)
        req1b = urlparse_cached(request1)
        req2 = urlparse_cached(request2)
        urlp = urlparse(url)

        assert req1a == req2
        assert req1a == urlp
        assert req1a is req1b
        assert req1a is not req2
        assert req1a is not req2


if __name__ == "__main__":
    unittest.main()

class MOff(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def __init__(self):
        raise NotConfigured


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


class GenspiderCommandTest(CommandTest):

    def test_arguments(self):
        # only pass one argument. spider script shouldn't be created
        self.assertEqual(2, self.call('genspider', 'test_name'))
        assert not exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))
        # pass two arguments <name> <domain>. spider script should be created
        self.assertEqual(0, self.call('genspider', 'test_name', 'test.com'))
        assert exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))

    def test_template(self, tplname='crawl'):
        args = ['--template=%s' % tplname] if tplname else []
        spname = 'test_spider'
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Created spider %r using template %r in module" % (spname, tplname), out)
        self.assertTrue(exists(join(self.proj_mod_path, 'spiders', 'test_spider.py')))
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Spider %r already exists in module" % spname, out)

    def test_template_basic(self):
        self.test_template('basic')

    def test_template_csvfeed(self):
        self.test_template('csvfeed')

    def test_template_xmlfeed(self):
        self.test_template('xmlfeed')

    def test_list(self):
        self.assertEqual(0, self.call('genspider', '--list'))

    def test_dump(self):
        self.assertEqual(0, self.call('genspider', '--dump=basic'))
        self.assertEqual(0, self.call('genspider', '-d', 'basic'))

    def test_same_name_as_project(self):
        self.assertEqual(2, self.call('genspider', self.project_name))
        assert not exists(join(self.proj_mod_path, 'spiders', '%s.py' % self.project_name))


class ChunkedTest(unittest.TestCase):

    def test_decode_chunked_transfer(self):
        """Example taken from: http://en.wikipedia.org/wiki/Chunked_transfer_encoding"""
        chunked_body = "25\r\n" + "This is the data in the first chunk\r\n\r\n"
        chunked_body += "1C\r\n" + "and this is the second one\r\n\r\n"
        chunked_body += "3\r\n" + "con\r\n"
        chunked_body += "8\r\n" + "sequence\r\n"
        chunked_body += "0\r\n\r\n"
        body = decode_chunked_transfer(chunked_body)
        self.assertEqual(body, \
            "This is the data in the first chunk\r\n" +
            "and this is the second one\r\n" +
            "consequence")



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


class TestItem(Item):
    name = Field()
    url = Field()


class FilesystemStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

class CrawlerLoggingTestCase(unittest.TestCase):
    def test_no_root_handler_installed(self):
        handler = get_scrapy_root_handler()
        if handler is not None:
            logging.root.removeHandler(handler)

        class MySpider(scrapy.Spider):
            name = 'spider'

        crawler = Crawler(MySpider, {})
        assert get_scrapy_root_handler() is None

    def test_spider_custom_settings_log_level(self):
        with tempfile.NamedTemporaryFile() as log_file:
            class MySpider(scrapy.Spider):
                name = 'spider'
                custom_settings = {
                    'LOG_LEVEL': 'INFO',
                    'LOG_FILE': log_file.name,
                }

            configure_logging()
            self.assertEqual(get_scrapy_root_handler().level, logging.DEBUG)
            crawler = Crawler(MySpider, {})
            self.assertEqual(get_scrapy_root_handler().level, logging.INFO)
            info_count = crawler.stats.get_value('log_count/INFO')
            logging.debug('debug message')
            logging.info('info message')
            logging.warning('warning message')
            logging.error('error message')
            logged = log_file.read().decode('utf8')
        self.assertNotIn('debug message', logged)
        self.assertIn('info message', logged)
        self.assertIn('warning message', logged)
        self.assertIn('error message', logged)
        self.assertEqual(crawler.stats.get_value('log_count/ERROR'), 1)
        self.assertEqual(crawler.stats.get_value('log_count/WARNING'), 1)
        self.assertEqual(
            crawler.stats.get_value('log_count/INFO') - info_count, 1)
        self.assertEqual(crawler.stats.get_value('log_count/DEBUG', 0), 0)


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


class MustbeDeferredTest(unittest.TestCase):
    def test_success_function(self):
        steps = []
        def _append(v):
            steps.append(v)
            return steps

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

    def test_unfired_deferred(self):
        steps = []
        def _append(v):
            steps.append(v)
            dfd = defer.Deferred()
            reactor.callLater(0, dfd.callback, steps)
            return dfd

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

def cb1(value, arg1, arg2):
    return "(cb1 %s %s %s)" % (value, arg1, arg2)
def cb2(value, arg1, arg2):
    return defer.succeed("(cb2 %s %s %s)" % (value, arg1, arg2))
def cb3(value, arg1, arg2):
    return "(cb3 %s %s %s)" % (value, arg1, arg2)
def cb_fail(value, arg1, arg2):
    return Failure(TypeError())
def eb1(failure, arg1, arg2):
    return "(eb1 %s %s %s)" % (failure.value.__class__.__name__, arg1, arg2)


def getPage(url, contextFactory=None, response_transform=None, *args, **kwargs):
    """Adapted version of twisted.web.client.getPage"""
    def _clientfactory(url, *args, **kwargs):
        url = to_unicode(url)
        timeout = kwargs.pop('timeout', 0)
        f = client.ScrapyHTTPClientFactory(
            Request(url, *args, **kwargs), timeout=timeout)
        f.deferred.addCallback(response_transform or (lambda r: r.body))
        return f

    from twisted.web.client import _makeGetterFactory
    return _makeGetterFactory(to_bytes(url), _clientfactory,
        contextFactory=contextFactory, *args, **kwargs).deferred


class DeprecatedImagesPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = mkdtemp()

    def init_pipeline(self, pipeline_class):
        self.pipeline = pipeline_class(self.tempdir, download_func=_mocked_download_func)
        self.pipeline.open_spider(None)

    def test_default_file_key_method(self):
        self.init_pipeline(ImagesPipeline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(self.pipeline.file_key("https://dev.mydeco.com/mydeco.gif"),
                             'full/3fd165099d8e71b8a48b2683946e64dbfad8b52d.jpg')
            self.assertEqual(len(w), 1)
            self.assertTrue('image_key(url) and file_key(url) methods are deprecated' in str(w[-1].message))

    def test_default_image_key_method(self):
        self.init_pipeline(ImagesPipeline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(self.pipeline.image_key("https://dev.mydeco.com/mydeco.gif"),
                             'full/3fd165099d8e71b8a48b2683946e64dbfad8b52d.jpg')
            self.assertEqual(len(w), 1)
            self.assertTrue('image_key(url) and file_key(url) methods are deprecated' in str(w[-1].message))

    def test_overridden_file_key_method(self):
        self.init_pipeline(DeprecatedImagesPipeline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(self.pipeline.file_path(Request("https://dev.mydeco.com/mydeco.gif")),
                             'empty/3fd165099d8e71b8a48b2683946e64dbfad8b52d.jpg')
            self.assertEqual(len(w), 1)
            self.assertTrue('image_key(url) and file_key(url) methods are deprecated' in str(w[-1].message))

    def test_default_thumb_key_method(self):
        self.init_pipeline(ImagesPipeline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(self.pipeline.thumb_key("file:///tmp/foo.jpg", 50),
                             'thumbs/50/38a86208c36e59d4404db9e37ce04be863ef0335.jpg')
            self.assertEqual(len(w), 1)
            self.assertTrue('thumb_key(url) method is deprecated' in str(w[-1].message))

    def test_overridden_thumb_key_method(self):
        self.init_pipeline(DeprecatedImagesPipeline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(self.pipeline.thumb_path(Request("file:///tmp/foo.jpg"), 50),
                             'thumbsup/50/38a86208c36e59d4404db9e37ce04be863ef0335.jpg')
            self.assertEqual(len(w), 1)
            self.assertTrue('thumb_key(url) method is deprecated' in str(w[-1].message))

    def tearDown(self):
        rmtree(self.tempdir)


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


class CustomSpiderLoader(SpiderLoader):
    pass


class ChunkedResource(resource.Resource):

    def render(self, request):
        def response():
            request.write(b"chunked ")
            request.write(b"content\n")
            request.finish()
        reactor.callLater(0, response)
        return server.NOT_DONE_YET


class LxmlLinkExtractorTestCase(Base.LinkExtractorTestCase):
    extractor_cls = LxmlLinkExtractor

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

    @pytest.mark.xfail
    def test_restrict_xpaths_with_html_entities(self):
        super(LxmlLinkExtractorTestCase, self).test_restrict_xpaths_with_html_entities()


class StatsCollectorTest(unittest.TestCase):

    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')

    def test_collector(self):
        stats = StatsCollector(self.crawler)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_value('anything'), None)
        self.assertEqual(stats.get_value('anything', 'default'), 'default')
        stats.set_value('test', 'value')
        self.assertEqual(stats.get_stats(), {'test': 'value'})
        stats.set_value('test2', 23)
        self.assertEqual(stats.get_stats(), {'test': 'value', 'test2': 23})
        self.assertEqual(stats.get_value('test2'), 23)
        stats.inc_value('test2')
        self.assertEqual(stats.get_value('test2'), 24)
        stats.inc_value('test2', 6)
        self.assertEqual(stats.get_value('test2'), 30)
        stats.max_value('test2', 6)
        self.assertEqual(stats.get_value('test2'), 30)
        stats.max_value('test2', 40)
        self.assertEqual(stats.get_value('test2'), 40)
        stats.max_value('test3', 1)
        self.assertEqual(stats.get_value('test3'), 1)
        stats.min_value('test2', 60)
        self.assertEqual(stats.get_value('test2'), 40)
        stats.min_value('test2', 35)
        self.assertEqual(stats.get_value('test2'), 35)
        stats.min_value('test4', 7)
        self.assertEqual(stats.get_value('test4'), 7)

    def test_dummy_collector(self):
        stats = DummyStatsCollector(self.crawler)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_value('anything'), None)
        self.assertEqual(stats.get_value('anything', 'default'), 'default')
        stats.set_value('test', 'value')
        stats.inc_value('v1')
        stats.max_value('v2', 100)
        stats.min_value('v3', 100)
        stats.open_spider('a')
        stats.set_value('test', 'value', spider=self.spider)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_stats('a'), {})

class SomeBaseClass(object):
    pass

