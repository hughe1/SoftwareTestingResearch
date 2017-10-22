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


def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/requests/requests/issues/3772.
    """
    text_200_chal = (b'HTTP/1.1 200 OK\r\n'
                     b'Content-Length: 0\r\n'
                     b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                     b', opaque="372825293d1c26955496c80ed6426e9e", '
                     b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to GET with a 200 containing www-authenticate header.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_200_chal)

        # Verify the client didn't respond with auth.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.get(url, auth=auth)
        # Verify server didn't receive auth from us.
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()


_schemes_by_var_prefix = [
    ('http', ['http']),
    ('https', ['https']),
    ('all', ['http', 'https']),
]




class TestLookupDict:

    @pytest.fixture(autouse=True)
    def setup(self):
        """LookupDict instance with "bad_gateway" attribute."""
        self.lookup_dict = LookupDict('test')
        self.lookup_dict.bad_gateway = 502

    def test_repr(self):
        assert repr(self.lookup_dict) == "<lookup 'test'>"

    get_item_parameters = pytest.mark.parametrize(
        'key, value', (
            ('bad_gateway', 502),
            ('not_a_key', None)
        )
    )

    @get_item_parameters
    def test_getitem(self, key, value):
        assert self.lookup_dict[key] == value

    @get_item_parameters
    def test_get(self, key, value):
        assert self.lookup_dict.get(key) == value

def test_default_hooks():
    assert hooks.default_hooks() == {'response': []}
import requests


@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}

