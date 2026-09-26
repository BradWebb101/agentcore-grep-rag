import * as path from 'node:path';
import * as cdk from 'aws-cdk-lib/core';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as agentcore from 'aws-cdk-lib/aws-bedrockagentcore';
import { Construct } from 'constructs';

/** Managed session storage mount. The agent copies the S3 corpus here, then greps it. */
export const SESSION_MOUNT = '/mnt/workspace';

/** Key prefix for the markdown dataset uploaded by BucketDeployment. */
export const CORPUS_PREFIX = 'corpus';

/** CloudWatch log group for runtime application and usage logs. */
export const RUNTIME_LOG_GROUP_NAME = '/aws/vendedlogs/bedrock-agentcore/grep_rag';

/** Bedrock foundation model. On-demand calls use a US or global inference profile, not this id. */
export const KIMI_K3_FOUNDATION_MODEL_ID = 'moonshotai.kimi-k3';

/**
 * Kimi K3 inference profile for a deploy region.
 * There is no in-region or EU profile. US and Canada use the US profile; every other region uses global.
 */
export function kimiK3InferenceProfileId(region: string): string {
  if (region.startsWith('us-') || region.startsWith('ca-')) {
    return 'us.moonshotai.kimi-k3';
  }
  return 'global.moonshotai.kimi-k3';
}

export interface GrepRagStackProps extends cdk.StackProps {
  /**
   * Bedrock model the agent calls from inside the runtime.
   * Enable Kimi K3 in the account before invoking the agent.
   * @default US or global inference profile for the stack region
   */
  readonly modelId?: string;
}

export class GrepRagStack extends cdk.Stack {
  readonly corpusBucket: s3.Bucket;
  readonly runtime: agentcore.Runtime;

  constructor(scope: Construct, id: string, props?: GrepRagStackProps) {
    super(scope, id, props);

    const modelId = props?.modelId ?? kimiK3InferenceProfileId(this.region);
    const foundationModelId = props?.modelId
      ? modelId.replace(/^(global|us|eu|au)\./, '')
      : KIMI_K3_FOUNDATION_MODEL_ID;

    this.corpusBucket = new s3.Bucket(this, 'CorpusBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
    });

    const runtimeLogGroup = new logs.LogGroup(this, 'RuntimeLogGroup', {
      logGroupName: RUNTIME_LOG_GROUP_NAME,
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.runtime = new agentcore.Runtime(this, 'GrepRagRuntime', {
      runtimeName: 'grep_rag',
      description: 'Copies an S3 corpus into session storage with the AWS CLI and answers by grepping those files.',
      agentRuntimeArtifact: agentcore.AgentRuntimeArtifact.fromAsset(path.join(__dirname, '../agent'), {
        platform: ecrAssets.Platform.LINUX_ARM64,
      }),
      loggingConfigs: [
        {
          logType: agentcore.LogType.APPLICATION_LOGS,
          destination: agentcore.LoggingDestination.cloudWatchLogs(runtimeLogGroup),
        },
        {
          logType: agentcore.LogType.USAGE_LOGS,
          destination: agentcore.LoggingDestination.cloudWatchLogs(runtimeLogGroup),
        },
      ],
      environmentVariables: {
        CORPUS_BUCKET: this.corpusBucket.bucketName,
        CORPUS_PREFIX,
        SESSION_MOUNT,
        MODEL_ID: modelId,
        LOG_LEVEL: 'INFO',
        PYTHONUNBUFFERED: '1',
        AWS_REGION: this.region,
        AWS_DEFAULT_REGION: this.region,
      },
    });

    new s3deploy.BucketDeployment(this, 'CorpusDeployment', {
      sources: [s3deploy.Source.asset(path.join(__dirname, '../corpus'))],
      destinationBucket: this.corpusBucket,
      destinationKeyPrefix: CORPUS_PREFIX,
      prune: true,
      memoryLimit: 512,
    });

    // RuntimeProps in this aws-cdk-lib does not expose filesystem configuration yet.
    // Session storage is a CloudFormation property on AWS::BedrockAgentCore::Runtime.
    const cfnRuntime = this.runtime.node.findChild('Resource') as agentcore.CfnRuntime;
    cfnRuntime.filesystemConfigurations = [
      {
        sessionStorage: { mountPath: SESSION_MOUNT },
      },
    ];

    // aws s3 cp needs GetObject plus ListBucket so a prefix copy can enumerate keys.
    this.corpusBucket.grantRead(this.runtime.role);

    this.runtime.addToRolePolicy(new iam.PolicyStatement({
      sid: 'InvokeGrepRagModel',
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream',
      ],
      resources: [
        `arn:${this.partition}:bedrock:*::foundation-model/${foundationModelId}`,
        `arn:${this.partition}:bedrock:${this.region}:${this.account}:inference-profile/${modelId}`,
      ],
    }));

    new cdk.CfnOutput(this, 'CorpusBucketName', {
      value: this.corpusBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'AgentRuntimeArn', {
      value: this.runtime.agentRuntimeArn,
    });
    new cdk.CfnOutput(this, 'SessionMount', {
      value: SESSION_MOUNT,
    });
    new cdk.CfnOutput(this, 'RuntimeLogGroupName', {
      value: runtimeLogGroup.logGroupName,
    });
  }
}
