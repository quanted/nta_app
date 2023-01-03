from datetime import datetime


def set_job_status(mongo, jobid, status, progress=0, n_masses="", create=False):
    posts = mongo.posts
    time_stamp = datetime.utcnow()
    post_id = jobid + "_" + "status"
    if create:
        posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                     'date': time_stamp,
                                                     'n_masses': str(n_masses),
                                                     'progress': str(progress),
                                                     'status': status,
                                                     'error_info': ''}},
                         upsert=True)
    else:
        posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                     'progress': str(progress),
                                                     'status': status}},
                         upsert=True)