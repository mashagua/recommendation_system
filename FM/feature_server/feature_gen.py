import redis
import pandas as pd
import traceback
import json
def save_redis(items, db=1):
    redis_url = 'redis://:test123@127.0.0.1:6379/'+str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()

# def save_redis(items, db=1):
#     redis_url = 'redis://test123@127.0.0.1:6379/'+str(db)
#     pool = redis.from_url(redis_url)
#     try:
#         for item in items:
#             pool.set(item[0], item[1])
#     except BaseException:
#         traceback.print_exc()


def get_user_feature():
    ds = pd.read_csv('../Data/click_log.csv')
    click_df = ds.sort_values(by='timestamp')
    user_environment_region_dict = {}
    for info in zip(
            click_df['user_id'],
            click_df['environment'],
            click_df['region']):
        user_environment_region_dict[info[0]] = (info[1], info[2])

    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['timestamp']))
    user_item_time_df = click_df.groupby('user_id')[
        'article_id', 'timestamp'].apply(
        lambda x: make_item_time_pair(x)).reset_index().rename(
            columns={
                0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    user_feature = []
    for user, item_time_dict in user_item_time_dict.items():
        info = user_environment_region_dict[user]
        tmp = (str(user),
               json.dumps({'user_id': user,
                           'hists': item_time_dict,
                           'environment': info[0],
                           'region': info[1]}))
        user_feature.append(tmp)

    save_redis(user_feature,1)


#
def get_item_feature():
    ds = pd.read_csv('../Data/articles.csv')
    ds = ds.to_dict(orient='records')
    item_feature = []
    for d in ds:
        item_feature.append((d['article_id'], json.dumps(d)))
    save_redis(item_feature, 2)

if __name__=='__main__':
    print('gen user feature ...')
    get_user_feature()
    print('gen item feature ...')
    get_item_feature()

