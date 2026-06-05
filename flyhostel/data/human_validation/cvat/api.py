import os
import logging
import datetime
import json
import requests

CVAT_BASE = "http://" + os.environ["CVAT_HOST"] + ":8080"
logger=logging.getLogger(__name__)

def cvat_auth(session):

    login_url = f"{CVAT_BASE}/api/auth/login"
    r = session.post(login_url, json={
        "username": os.environ["CVAT_USERNAME"],
        "password": os.environ["CVAT_PASSWORD"]
    })
    r.raise_for_status()
    return r



def delete_task(task_id: int) -> None:
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

    with requests.Session() as s:
        # 1) Log in (endpoint and payload depend on your API)
        r = cvat_auth(s)
    
        # 2) Now cookies are stored in `s`, and will be sent automatically
        url = f"{CVAT_BASE}/api/tasks"
        r = s.get(url, params={"project_id": project_id})
        r.raise_for_status()
        out = r.json()
        tasks=[]
        for task in out["results"]:
            tasks.append(int(task["id"]))
        return tuple(tasks)
    


def delete_task_annotations(task_number):
    url = f"http://localhost:8080/api/tasks/{task_number}/annotations/"
    print(f"Fetching {url}")

    with requests.Session() as s:
        cvat_auth(s)

        # Pull the CSRF token that login set, and echo it back as a header
        csrf = s.cookies.get("csrftoken")
        headers = {
            "X-CSRFTOKEN": csrf,
            "Referer": "http://localhost:8080",  # DRF also checks this on unsafe methods
        }

        r = s.delete(url, headers=headers)
        r.raise_for_status()
        print(f"Deleted annotations for task {task_number} (status {r.status_code})")

    
def get_task_mtime(task_number):

    """
    
    Arguments
        task (int):
    """
    url=f"http://localhost:8080/api/tasks/{task_number}"
    print(f"Fetching {url}")

    with requests.Session() as s:
        # 1) Log in (endpoint and payload depend on your API)
        r = cvat_auth(s)
    
        # 2) Now cookies are stored in `s`, and will be sent automatically
        r = s.get(url)
        r.raise_for_status()
        out = r.json()

        with open(PROJECTS_JSON, 'w') as handle:
            json.dump(out, handle)

    updated_date_without_ms=out["updated_date"].split(".")[0]
    dt = datetime.datetime.fromisoformat(updated_date_without_ms)
    return dt 

PROJECTS_JSON="/flyhostel_data/videos/index_cvat_projects.json"
def update_project_list():
    url="http://localhost:8080/api/projects?page_size=9999&scheme=json"
    print(f"Fetching {url}")

    with requests.Session() as s:
        # 1) Log in (endpoint and payload depend on your API)
        r = cvat_auth(s)
    
        # 2) Now cookies are stored in `s`, and will be sent automatically
        r = s.get(url)
        r.raise_for_status()
        out = r.json()

        with open(PROJECTS_JSON, 'w') as handle:
            json.dump(out, handle)

    return out



def get_project_id_from_name(experiment, errors="raise"):

    update_project_list()

    with open(PROJECTS_JSON, "r") as handle:
        index_cvat_projects=json.load(handle)
    
    assert index_cvat_projects["next"] is None
    assert index_cvat_projects["previous"] is None
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
