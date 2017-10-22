# -*- coding: utf-8 -*-

"""Tests for Requests."""

from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings

import io
import requests
import pytest
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
    Morsel, cookielib, getproxies, str, urlparse,
    builtin_str, OrderedDict)
from requests.cookies import (
    cookiejar_from_dict, morsel_to_cookie)
from requests.exceptions import (
    ConnectionError, ConnectTimeout, InvalidSchema, InvalidURL,
    MissingSchema, ReadTimeout, Timeout, RetryError, TooManyRedirects,
    ProxyError, InvalidHeader, UnrewindableBodyError, SSLError)
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks

# from .compat import StringIO, u
# from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout

# Requests to this URL should always fail with a connection timeout (nothing
# listening on that port)
TARPIT = 'http://10.255.255.1'

try:
    from ssl import SSLContext
    del SSLContext
    HAS_MODERN_SSL = True
except ImportError:
    HAS_MODERN_SSL = False

try:
    requests.pyopenssl
    HAS_PYOPENSSL = True
except AttributeError:
    HAS_PYOPENSSL = False

# -*- coding: utf-8 -*-

import threading
import socket
import time

import pytest
import requests
from testserver.server import Server

# -*- coding: utf-8 -*-

import pytest

from requests.structures import CaseInsensitiveDict, LookupDict

# -*- coding: utf-8 -*-

import os
import copy
from io import BytesIO

import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
    address_in_network, dotted_netmask,
    get_auth_from_url, get_encoding_from_headers,
    get_encodings_from_content, get_environ_proxies,
    guess_filename, guess_json_utf, is_ipv4_address,
    is_valid_cidr, iter_slices, parse_dict_header,
    parse_header_links, prepend_scheme_if_needed,
    requote_uri, select_proxy, should_bypass_proxies, super_len,
    to_key_val_list, to_native_string,
    unquote_header_value, unquote_unreserved,
    urldefragauth, add_dict_to_cookiejar, set_environ)
from requests._internal_utils import unicode_is_ascii


# -*- encoding: utf-8

import sys

import pytest

from requests.help import info


class TestCaseInsensitiveDict:

    @pytest.fixture(autouse=True)
    def setup(self):
        """CaseInsensitiveDict instance with "Accept" header."""
        self.case_insensitive_dict = CaseInsensitiveDict()
        self.case_insensitive_dict['Accept'] = 'application/json'

    def test_list(self):
        assert list(self.case_insensitive_dict) == ['Accept']

    possible_keys = pytest.mark.parametrize('key', ('accept', 'ACCEPT', 'aCcEpT', 'Accept'))

    @possible_keys
    def test_getitem(self, key):
        assert self.case_insensitive_dict[key] == 'application/json'

    @possible_keys
    def test_delitem(self, key):
        del self.case_insensitive_dict[key]
        assert key not in self.case_insensitive_dict

    def test_lower_items(self):
        assert list(self.case_insensitive_dict.lower_items()) == [('accept', 'application/json')]

    def test_repr(self):
        assert repr(self.case_insensitive_dict) == "{'Accept': 'application/json'}"

    def test_copy(self):
        copy = self.case_insensitive_dict.copy()
        assert copy is not self.case_insensitive_dict
        assert copy == self.case_insensitive_dict

    @pytest.mark.parametrize(
        'other, result', (
            ({'AccePT': 'application/json'}, True),
            ({}, False),
            (None, False)
        )
    )
    def test_instance_equality(self, other, result):
        assert (self.case_insensitive_dict == other) is result


class TestGetEnvironProxies:
    """Ensures that IP addresses are correctly matches with ranges
    in no_proxy variable.
    """

    @pytest.fixture(autouse=True, params=['no_proxy', 'NO_PROXY'])
    def no_proxy(self, request, monkeypatch):
        monkeypatch.setenv(request.param, '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.0.1:5000/',
            'http://192.168.0.1/',
            'http://172.16.1.1/',
            'http://172.16.1.1:5000/',
            'http://localhost.localdomain:5000/v1.0/',
        ))
    def test_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) == {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.1.1:5000/',
            'http://192.168.1.1/',
            'http://www.requests.com/',
        ))
    def test_not_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) != {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.1.1:5000/',
            'http://192.168.1.1/',
            'http://www.requests.com/',
        ))
    def test_bypass_no_proxy_keyword(self, url):
        no_proxy = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) == {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.0.1:5000/',
            'http://192.168.0.1/',
            'http://172.16.1.1/',
            'http://172.16.1.1:5000/',
            'http://localhost.localdomain:5000/v1.0/',
        ))
    def test_not_bypass_no_proxy_keyword(self, url, monkeypatch):
        # This is testing that the 'no_proxy' argument overrides the
        # environment variable 'no_proxy'
        monkeypatch.setenv('http_proxy', 'http://proxy.example.com:3128/')
        no_proxy = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) != {}


class TestAddressInNetwork:

    def test_valid(self):
        assert address_in_network('192.168.1.1', '192.168.1.0/24')

    def test_invalid(self):
        assert not address_in_network('172.16.0.1', '192.168.1.0/24')


def test_can_access_urllib3_attribute():
    requests.packages.urllib3


class TestMorselToCookieMaxAge:

    """Tests for morsel_to_cookie when morsel contains max-age."""

    def test_max_age_valid_int(self):
        """Test case where a valid max age in seconds is passed."""

        morsel = Morsel()
        morsel['max-age'] = 60
        cookie = morsel_to_cookie(morsel)
        assert isinstance(cookie.expires, int)

    def test_max_age_invalid_str(self):
        """Test case where a invalid max age is passed."""

        morsel = Morsel()
        morsel['max-age'] = 'woops'
        with pytest.raises(TypeError):
            morsel_to_cookie(morsel)

