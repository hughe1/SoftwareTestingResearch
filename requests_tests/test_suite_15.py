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


def test_can_access_urllib3_attribute():
    requests.packages.urllib3


def test_redirect_rfc1808_to_non_ascii_location():
    path = u'š'
    expected_path = b'%C5%A1'
    redirect_request = []  # stores the second request to the server

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = u'//{0}:{1}/{2}'.format(host, port, path)
        sock.send(
            b'HTTP/1.1 301 Moved Permanently\r\n'
            b'Content-Length: 0\r\n'
            b'Location: ' + location.encode('utf8') + b'\r\n'
            b'\r\n'
        )
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')

    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = u'http://{0}:{1}'.format(host, port)
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == u'{0}/{1}'.format(url, expected_path.decode('ascii'))

        close_server.set()

def test_default_hooks():
    assert hooks.default_hooks() == {'response': []}

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


@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}

