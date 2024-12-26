from django.conf import settings
from django.contrib.auth import login as django_login
from django.template.loader import render_to_string
from django.http import HttpResponse
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.http import HttpResponseForbidden
import re
import os
import logging
import subprocess

# Set up logging
logger = logging.getLogger("nta_app.login-middleware")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class Http403Middleware(object):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if HttpResponseForbidden.status_code == response.status_code:
            logger.info("User login session token timed out")
            return login(
                request,
                "<span style='color:red;'>Your session has timed out, please log back in to refresh your session.</span>",
            )
        else:
            return response


def login(request):
    next_page = request.GET.get("next")
    message = ""
    delete_message = False
    if "message" in request.COOKIES.keys():
        message = request.COOKIES["message"]
        delete_message = True
    html = render_to_string(
        "login_prompt.html",
        {"TITLE": "User Login", "next": next_page, "TEXT": message},
        request=request,
    )
    response = HttpResponse()
    response.write(html)
    if delete_message:
        response.delete_cookie("message")
    return response


class RequireLoginMiddleware:
    def __init__(self, get_response):

        self.login_verbose = settings.LOGIN_VERBOSE
        self.login_duration = settings.LOGIN_DURATION

        if not settings.LOGIN_REQUIRED:
            return

        self.get_response = get_response
        self.login_url = re.compile(settings.LOGIN_URL)
        if os.getenv("DEPLOY_ENV", "kube-dev") == "kube-prod":  # set username based on deploy
            self.nta_username = "ntauser"
        else:
            self.nta_username = "ntadev"
        self.open_urls = ["/nta/login", "/external/", "/processing/", "/status/", "/results/"]

        nta_password = self.load_password()
        if nta_password is None:
            logger.warning("NTA login password as not set.")
            return

        try:
            if not User.objects.filter(username=self.nta_username).exists():
                _user = User.objects.create_user(self.nta_username, "nta@nta.nta", nta_password)
                _user.save()
        except Exception:
            logger.warning(f"User: {self.nta_username} already exists")

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def load_password(self):
        nta_password = os.getenv("NTA_PASSWORD")
        if nta_password is None:
            if self.login_verbose:
                logger.warning(f"Unable to get password as NTA_PASSWORD env was not set.")
            return None

        reset_env = ""
        if os.name == "posix":
            reset_env = "export NTA_PASSWORD=THEANSWERIS42"
        elif os.name == "nt":
            reset_env = "setx NTA_PASSWORD THEANSWERIS42"
        subprocess.Popen(reset_env, shell=True).wait()
        return nta_password

    def login_auth(self, request):

        username = request.POST.get("username")
        password = request.POST.get("password")
        next_page = request.POST.get("next")

        # Redirect back to login page if username is invalid
        if username != self.nta_username:
            if self.login_verbose:
                logger.info(f"Login Auth: Username is invalid. Provided username: {username}")
            response = redirect("/nta/login?next={}".format(next_page))
            response.set_cookie("message", "Username is not correct.")
            return response

        user = authenticate(username=username, password=password)
        session_duration = self.login_duration
        if user is not None:
            if user.is_active:
                if self.login_verbose:
                    logger.info(
                        f"User login successful, redirecting to {next_page}, session will expire in {session_duration} secs."
                    )
                request.session.set_expiry(session_duration)
                django_login(request, user)
                return redirect(next_page)
            else:
                if self.login_verbose:
                    logger.info(f"User no longer active, must log back in.")
                response = redirect("/nta/login?next={}".format(next_page))
                response.set_cookie("message", "Session has ended, must log back in.")
                return response
        else:
            if self.login_verbose:
                logger.info(f"User not logged in.")
            response = redirect("/nta/login?next={}".format(next_page))
            response.set_cookie("message", "Password is not correct.")
            return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        assert hasattr(request, "user")
        path = request.path
        redirect_path = request.POST.get("next", "")
        token = request.GET.get("token")
        user = request.user
        if self.login_verbose:
            logger.debug(f"Process view; user: {user}, next: {redirect_path}, token: {token}")
        if request.POST and self.login_url.match(path):
            return self.login_auth(request)
        elif not user.is_authenticated:
            if self.open_url_check(path=path):
                if self.login_verbose:
                    logger.debug(f"Process view cleared open url for path: {path}")
                return
            else:
                return redirect("/nta/login?next={}".format(path))
        elif user.is_authenticated:
            return
        else:
            return redirect("/nta/login?next={}".format(path))

    def open_url_check(self, path):
        if any(p in path for p in self.open_urls):
            return True
        return False
