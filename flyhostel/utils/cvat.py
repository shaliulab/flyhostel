import sqlite3
import glob
import os
import logging
import datetime
import json
import requests
from flyhostel.data.human_validation.constants import CVAT_HOST, _require_cvat
CVAT_BASE = "http://" + CVAT_HOST + ":8080"
logger=logging.getLogger(__name__)


def cvat_auth(session):

    _require_cvat()

    login_url = f"{CVAT_BASE}/api/auth/login"
    r = session.post(login_url, json={
        "username": os.environ["CVAT_USERNAME"],
        "password": os.environ["CVAT_PASSWORD"]
    })
    r.raise_for_status()
    return r



def delete_task(task_id: int) -> None:

    _require_cvat()

    url = f"{CVAT_BASE}/api/tasks/{task_id}"
    with requests.Session() as s:
        cvat_auth(s)

        # CVAT/Django CSRF: cookie name is typically "csrftoken"
        csrf = s.cookies.get("csrftoken")
        headers = {}
        if csrf:
            headers["X-CSRFToken"] = csrf
            headers["Referer"] = CVAT_BASE  # often required in CSRF checks

        r = s.delete(url, headers=headers)

        if r.status_code == 403:
            # This will usually say something like "CSRF Failed: CSRF token missing."
            raise RuntimeError(f"403 deleting task {task_id}: {r.text}")

        r.raise_for_status()
        # 204 => no JSON to parse
        return None



def get_tasks_for_project(project_id):

    _require_cvat()
    url = f"{CVAT_BASE}/api/tasks"

    with requests.Session() as s:
        # 1) Log in (endpoint and payload depend on your API)
        r = cvat_auth(s)
    
        # 2) Now cookies are stored in `s`, and will be sent automatically
        r = s.get(url, params={"project_id": project_id})
        r.raise_for_status()
        out = r.json()
        tasks=[]
        for task in out["results"]:
            tasks.append(int(task["id"]))
        return tuple(tasks)
    


def delete_task_annotations(task_number):

    _require_cvat()

    url = f"{CVAT_BASE}/api/tasks/{task_number}/annotations/"
    print(f"Fetching {url}")

    with requests.Session() as s:
        cvat_auth(s)

        # Pull the CSRF token that login set, and echo it back as a header
        csrf = s.cookies.get("csrftoken")
        headers = {
            "X-CSRFTOKEN": csrf,
            "Referer": CVAT_BASE,  # DRF also checks this on unsafe methods
        }

        r = s.delete(url, headers=headers)
        r.raise_for_status()
        print(f"Deleted annotations for task {task_number} (status {r.status_code})")

    
def get_task_mtime(task_number):

    """
    
    Arguments
        task (int):
    """
    _require_cvat()

    url=f"{CVAT_BASE}/api/tasks/{task_number}"
    print(f"Fetching {url}")

    with requests.Session() as s:
        # 1) Log in (endpoint and payload depend on your API)
        r = cvat_auth(s)
    
        # 2) Now cookies are stored in `s`, and will be sent automatically
        r = s.get(url)
        r.raise_for_status()
        out = r.json()

    updated_date_without_ms=out["updated_date"].split(".")[0]
    dt = datetime.datetime.fromisoformat(updated_date_without_ms)
    return dt 

PROJECTS_JSON="/flyhostel_data/videos/index_cvat_projects.json"

def file_is_older_than_seconds(path, seconds):
    dt = datetime.datetime.fromtimestamp(
        os.path.getmtime(path)
    )
    return (dt + datetime.timedelta(seconds=seconds)) < datetime.datetime.now()

def update_project_list():
    _require_cvat()

    if not os.path.exists(PROJECTS_JSON) or file_is_older_than_seconds(PROJECTS_JSON, 60):
        url=f"{CVAT_BASE}/api/projects?page_size=9999&scheme=json"
        print(f"Fetching {url}")

        with requests.Session() as s:
            # 1) Log in (endpoint and payload depend on your API)
            r = cvat_auth(s)
        
            # 2) Now cookies are stored in `s`, and will be sent automatically
            r = s.get(url)
            r.raise_for_status()
            out = r.json()

            if not os.path.exists(PROJECTS_JSON) or file_is_older_than_seconds(PROJECTS_JSON, 60):
                with open(PROJECTS_JSON, 'w') as handle:
                    json.dump(out, handle)
    else:
        with open(PROJECTS_JSON, 'r') as handle:
            out=json.load(handle)

    return out



def get_project_id_from_name(experiment, errors="raise"):

    _require_cvat()

    update_project_list()

    with open(PROJECTS_JSON, "r") as handle:
        index_cvat_projects=json.load(handle)
    
    assert index_cvat_projects.get("next", None) is None
    assert index_cvat_projects.get("previous", None) is None
    project_id=None
    hit=False
    for project in index_cvat_projects["results"]:
        if project["name"]==experiment:
            if hit == True:
                if errors=="raise":
                    raise Exception(f"More than 1 project with the same name (experiment)")
                elif errors=="ignore":
                    logger.warning("More than 1 project with the same name %s", experiment)
            
            project_id=project["id"]
            hit=True
    
    if project_id is None:
        if errors=="raise":
            raise Exception(f"0 projects with name {experiment}")
        elif errors=="ignore":
            logger.warning("0 projects with name %s", experiment)

    return project_id

    

def get_experiment_identifier(basedir):
    return "_".join(basedir.rstrip(os.path.sep).split(os.path.sep)[-3:])

def get_dbfile(basedir):
    dbfile=os.path.join(
        basedir,
        get_experiment_identifier(basedir) + ".db"
    )
    assert os.path.exists(dbfile), f"{dbfile} not found"
    return dbfile

def get_basedir(experiment):
    tokens = experiment.split("_")
    basedir=f"{os.environ['FLYHOSTEL_VIDEOS']}/{tokens[0]}/{tokens[1]}/{'_'.join(tokens[2:4])}"
    return basedir


def experiment_is_validated(experiment, errors="ignore"):
    basedir=get_basedir(experiment)
    dbfile=get_dbfile(basedir)
    
    annotation_files=glob.glob(f"{basedir}/flyhostel/validation/*_annotations.zip")
    if len(annotation_files)>0:
        return True
    else:

        with sqlite3.connect(dbfile) as conn:
            cur=conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables=cur.fetchall()
            tables=[e[0] for e in tables]
        if "ROI_VAL" in tables and "IDENTITY_VAL" in tables:
            return True
        else:
            project_id=get_project_id_from_name(experiment, errors=errors)
            return project_id is not None