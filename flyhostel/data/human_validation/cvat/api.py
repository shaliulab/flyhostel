import os

import requests

CVAT_BASE = "http://" + os.environ["CVAT_HOST"] + ":8080"


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
        return tasks